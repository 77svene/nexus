"""
modules/vram_orchestrator.py

Predictive Resource Orchestration for nexus
AI-driven prediction of resource needs based on prompt analysis
"""

import os
import re
import json
import time
import pickle
import hashlib
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import gc

from modules import shared, sd_models, sd_vae, extra_networks, scripts, devices
from modules.paths import models_path
from modules.shared import opts
from modules.processing import StableDiffusionProcessing
from modules.sd_models import model_hash
from modules.lora import network as lora_network


@dataclass
class ResourcePrediction:
    """Prediction result for resource allocation"""
    checkpoint: Optional[str] = None
    vae: Optional[str] = None
    loras: List[str] = None
    textual_inversions: List[str] = None
    upscaler: Optional[str] = None
    adetailer_models: List[str] = None
    estimated_vram_mb: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.loras is None:
            self.loras = []
        if self.textual_inversions is None:
            self.textual_inversions = []
        if self.adetailer_models is None:
            self.adetailer_models = []
        self.timestamp = time.time()


@dataclass
class ResourceUsage:
    """Actual resource usage after generation"""
    checkpoint: str
    vae: str
    loras: List[str]
    textual_inversions: List[str]
    upscaler: Optional[str]
    adetailer_models: List[str]
    peak_vram_mb: float
    generation_time: float
    prompt_hash: str
    timestamp: float


class PromptFeatureExtractor:
    """Extract features from prompts for prediction"""
    
    # Common model keywords and patterns
    CHECKPOINT_PATTERNS = [
        r'sd[12]\\.?[0-9]', r'stable.?diffusion', r'deliberate', r'realistic.?vision',
        r'dreamshaper', r'anythingv[0-9]', r'chilloutmix', r'openjourney',
        r'protogen', r'f222', r'hassan', r'cyberrealistic', r'photon'
    ]
    
    LORA_PATTERNS = [
        r'<lora:([^:>]+)(?::[^>]+)?>',  # <lora:name:weight>
        r'lora[_\\-]?([a-z0-9_]+)',  # lora_name
        r'lyco[_\\-]?([a-z0-9_]+)',  # lyco_name
    ]
    
    UPSCALER_PATTERNS = [
        r'4x', r'upscale', r'upsampling', r'enhance', r'high.?res',
        r'ldsr', r'esrgan', r'real.?ersgan', r'swinir', r'latent'
    ]
    
    ADETAILER_PATTERNS = [
        r'adetailer', r'face.?fix', r'face.?restore', r'after.?detailer',
        r'face.?only', r'eye.?detail', r'hand.?detail'
    ]
    
    def __init__(self):
        self.keyword_weights = self._load_keyword_weights()
        
    def _load_keyword_weights(self) -> Dict[str, float]:
        """Load keyword weights from config or use defaults"""
        config_path = Path(models_path) / "vram_orchestrator" / "keyword_weights.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default weights
        return {
            'checkpoint': 0.7,
            'lora': 0.6,
            'upscaler': 0.8,
            'adetailer': 0.5,
            'high_res': 0.9,
            'batch_size': 0.4,
            'steps': 0.3
        }
    
    def extract_features(self, p: StableDiffusionProcessing) -> Dict[str, Any]:
        """Extract features from processing object"""
        prompt = p.prompt if hasattr(p, 'prompt') else ""
        negative_prompt = p.negative_prompt if hasattr(p, 'negative_prompt') else ""
        
        features = {
            'prompt_length': len(prompt.split()),
            'negative_prompt_length': len(negative_prompt.split()),
            'has_lora': bool(re.search(r'<lora:[^>]+>', prompt, re.IGNORECASE)),
            'has_lyco': bool(re.search(r'<lyco:[^>]+>', prompt, re.IGNORECASE)),
            'has_upscaler': any(re.search(pattern, prompt, re.IGNORECASE) 
                               for pattern in self.UPSCALER_PATTERNS),
            'has_adetailer': any(re.search(pattern, prompt, re.IGNORECASE) 
                                for pattern in self.ADETAILER_PATTERNS),
            'width': getattr(p, 'width', 512),
            'height': getattr(p, 'height', 512),
            'batch_size': getattr(p, 'batch_size', 1),
            'steps': getattr(p, 'steps', 20),
            'cfg_scale': getattr(p, 'cfg_scale', 7.0),
            'sampler_name': getattr(p, 'sampler_name', 'Euler a'),
            'seed': getattr(p, 'seed', -1),
            'subseed': getattr(p, 'subseed', -1),
            'prompt_keywords': self._extract_keywords(prompt),
            'negative_keywords': self._extract_keywords(negative_prompt)
        }
        
        # Extract specific model names
        features['checkpoint_hint'] = self._extract_checkpoint_hint(prompt)
        features['lora_names'] = self._extract_lora_names(prompt)
        features['upscaler_hint'] = self._extract_upscaler_hint(prompt)
        features['adetailer_hint'] = self._extract_adetailer_hint(prompt)
        
        return features
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove special tokens and common words
        text = re.sub(r'<[^>]+>', '', text)  # Remove special tokens
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        
        words = text.lower().split()
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Return top 20 keywords by frequency
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(20)]
    
    def _extract_checkpoint_hint(self, prompt: str) -> Optional[str]:
        """Try to extract checkpoint name from prompt"""
        for pattern in self.CHECKPOINT_PATTERNS:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return None
    
    def _extract_lora_names(self, prompt: str) -> List[str]:
        """Extract LoRA names from prompt"""
        names = []
        
        # Standard LoRA syntax
        for pattern in self.LORA_PATTERNS:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            names.extend(matches)
        
        # Also check for common naming patterns in prompt
        lora_keywords = ['lora', 'lyco', 'locon', 'dylora']
        words = prompt.lower().split()
        for i, word in enumerate(words):
            if any(keyword in word for keyword in lora_keywords):
                # Look for potential model names nearby
                context = ' '.join(words[max(0, i-2):min(len(words), i+3)])
                # Extract alphanumeric sequences that might be model names
                potential_names = re.findall(r'[a-z0-9_]{3,}', context)
                names.extend(potential_names)
        
        return list(set(names))  # Remove duplicates
    
    def _extract_upscaler_hint(self, prompt: str) -> Optional[str]:
        """Extract upscaler hint from prompt"""
        for pattern in self.UPSCALER_PATTERNS:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return None
    
    def _extract_adetailer_hint(self, prompt: str) -> Optional[str]:
        """Extract adetailer hint from prompt"""
        for pattern in self.ADETAILER_PATTERNS:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return None


class PredictionModel:
    """Lightweight model for predicting resource needs"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or Path(models_path) / "vram_orchestrator" / "prediction_model.pkl"
        self.feature_extractor = PromptFeatureExtractor()
        self.model = None
        self.feature_history = []
        self.usage_history = []
        self.max_history = 1000
        self.model_version = "1.0.0"
        
        # Load or initialize model
        self._load_model()
    
    def _load_model(self):
        """Load prediction model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('version') == self.model_version:
                        self.model = data.get('model')
                        self.feature_history = data.get('history', [])[-self.max_history:]
                        self.usage_history = data.get('usage', [])[-self.max_history:]
                        print(f"[VRAM Orchestrator] Loaded prediction model with {len(self.feature_history)} samples")
            except Exception as e:
                print(f"[VRAM Orchestrator] Failed to load model: {e}")
                self._initialize_model()
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize a simple rule-based model"""
        # Simple rule-based model as fallback
        self.model = {
            'type': 'rule_based',
            'checkpoint_rules': {
                'realistic': ['realistic', 'photo', 'portrait', 'face'],
                'anime': ['anime', 'manga', '2d', 'cartoon'],
                'sd2': ['sd2', 'stable diffusion 2', 'v2'],
                'sd_xl': ['sdxl', 'sd_xl', 'xl', 'extra-large']
            },
            'lora_rules': {
                'character': ['character', 'person', 'girl', 'boy', 'man', 'woman'],
                'style': ['style', 'artistic', 'painting', 'sketch'],
                'clothing': ['outfit', 'dress', 'clothing', 'armor']
            },
            'upscaler_rules': {
                '4x': ['4x', 'upscale', 'high-res', 'high resolution'],
                'face': ['face', 'detail', 'restore', 'fix']
            }
        }
        print("[VRAM Orchestrator] Initialized rule-based prediction model")
    
    def _save_model(self):
        """Save prediction model to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'version': self.model_version,
                    'model': self.model,
                    'history': self.feature_history,
                    'usage': self.usage_history,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            print(f"[VRAM Orchestrator] Failed to save model: {e}")
    
    def predict(self, p: StableDiffusionProcessing, aggressiveness: float = 0.5) -> ResourcePrediction:
        """Predict resources needed for generation"""
        features = self.feature_extractor.extract_features(p)
        
        if self.model['type'] == 'rule_based':
            return self._predict_rule_based(features, aggressiveness)
        else:
            # Future: ML-based prediction
            return self._predict_rule_based(features, aggressiveness)
    
    def _predict_rule_based(self, features: Dict[str, Any], aggressiveness: float) -> ResourcePrediction:
        """Rule-based prediction"""
        prediction = ResourcePrediction()
        
        # Predict checkpoint
        prompt_text = ' '.join(features['prompt_keywords']).lower()
        checkpoint_hint = features.get('checkpoint_hint', '')
        
        if checkpoint_hint:
            for category, keywords in self.model['checkpoint_rules'].items():
                if any(keyword in checkpoint_hint for keyword in keywords):
                    # Try to find matching checkpoint
                    matching_checkpoints = self._find_matching_checkpoints(category)
                    if matching_checkpoints:
                        prediction.checkpoint = matching_checkpoints[0]
                        break
        
        # Predict LoRAs
        if features['has_lora'] or features['has_lyco']:
            lora_names = features.get('lora_names', [])
            if lora_names:
                # Find matching LoRAs in model directory
                matching_loras = self._find_matching_loras(lora_names)
                prediction.loras = matching_loras[:3]  # Limit to 3 LoRAs
        
        # Predict upscaler
        if features['has_upscaler']:
            upscaler_hint = features.get('upscaler_hint', '')
            if '4x' in upscaler_hint or 'upscale' in upscaler_hint:
                prediction.upscaler = "4x-UltraSharp"  # Default upscaler
            elif 'face' in upscaler_hint:
                prediction.upscaler = "CodeFormer"
        
        # Predict adetailer
        if features['has_adetailer']:
            adetailer_hint = features.get('adetailer_hint', '')
            if 'face' in adetailer_hint:
                prediction.adetailer_models.append("face_yolov8n.pt")
            if 'hand' in adetailer_hint:
                prediction.adetailer_models.append("hand_yolov8n.pt")
        
        # Estimate VRAM usage
        prediction.estimated_vram_mb = self._estimate_vram(features, prediction)
        
        # Adjust confidence based on aggressiveness
        prediction.confidence = min(1.0, aggressiveness * 0.8 + 0.2)
        
        return prediction
    
    def _find_matching_checkpoints(self, category: str) -> List[str]:
        """Find checkpoints matching a category"""
        matching = []
        checkpoint_info = sd_models.checkpoint_info
        
        if checkpoint_info:
            # Check current checkpoint name
            name_lower = checkpoint_info.name.lower()
            if category in name_lower:
                matching.append(checkpoint_info.name)
        
        # Also check available checkpoints
        for info in sd_models.checkpoints_list.values():
            if category in info.name.lower():
                matching.append(info.name)
        
        return list(set(matching))[:5]  # Return top 5 unique matches
    
    def _find_matching_loras(self, lora_names: List[str]) -> List[str]:
        """Find LoRAs matching the extracted names"""
        matching = []
        
        # Get available LoRAs
        available_loras = []
        try:
            from modules import extra_networks
            lora_path = Path(models_path) / "Lora"
            if lora_path.exists():
                for file in lora_path.rglob("*.safetensors"):
                    available_loras.append(file.stem)
                for file in lora_path.rglob("*.pt"):
                    available_loras.append(file.stem)
        except:
            pass
        
        # Find matches
        for name in lora_names:
            name_lower = name.lower()
            for available in available_loras:
                if name_lower in available.lower() or available.lower() in name_lower:
                    matching.append(available)
                    break
        
        return list(set(matching))  # Remove duplicates
    
    def _estimate_vram(self, features: Dict[str, Any], prediction: ResourcePrediction) -> float:
        """Estimate VRAM usage in MB"""
        base_vram = 1500  # Base VRAM for SD 1.5
        
        # Adjust for resolution
        width = features.get('width', 512)
        height = features.get('height', 512)
        pixels = width * height
        resolution_factor = pixels / (512 * 512)
        
        # Adjust for batch size
        batch_size = features.get('batch_size', 1)
        
        # Adjust for steps
        steps = features.get('steps', 20)
        steps_factor = steps / 20
        
        # Calculate estimated VRAM
        estimated = base_vram * resolution_factor * batch_size * steps_factor
        
        # Add overhead for LoRAs
        estimated += len(prediction.loras) * 200
        
        # Add overhead for upscaler
        if prediction.upscaler:
            estimated += 500
        
        # Add overhead for adetailer
        estimated += len(prediction.adetailer_models) * 300
        
        return estimated
    
    def update(self, features: Dict[str, Any], usage: ResourceUsage):
        """Update model with new data"""
        self.feature_history.append(features)
        self.usage_history.append(asdict(usage))
        
        # Trim history if too long
        if len(self.feature_history) > self.max_history:
            self.feature_history = self.feature_history[-self.max_history:]
            self.usage_history = self.usage_history[-self.max_history:]
        
        # Periodically save model
        if len(self.feature_history) % 100 == 0:
            self._save_model()


class VRAMOrchestrator:
    """Main orchestrator for predictive resource management"""
    
    def __init__(self):
        self.enabled = True
        self.aggressiveness = 0.5  # 0.0 to 1.0
        self.preload_enabled = True
        self.unload_threshold = 0.8  # Unload when VRAM usage > 80%
        self.prediction_horizon = 2  # Predict 2 steps ahead
        self.model_cache_size = 3  # Keep 3 models in cache
        
        self.prediction_model = PredictionModel()
        self.resource_cache = {}
        self.vram_monitor = VRAMMonitor()
        self.preload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vram_preload")
        self.pending_predictions: Dict[str, Future] = {}
        
        # Load settings
        self._load_settings()
        
        # Start background monitoring
        self._start_monitoring()
    
    def _load_settings(self):
        """Load orchestrator settings from opts"""
        self.enabled = getattr(opts, 'vram_orchestrator_enabled', True)
        self.aggressiveness = getattr(opts, 'vram_orchestrator_aggressiveness', 0.5)
        self.preload_enabled = getattr(opts, 'vram_orchestrator_preload', True)
        self.unload_threshold = getattr(opts, 'vram_orchestrator_unload_threshold', 0.8)
        self.prediction_horizon = getattr(opts, 'vram_orchestrator_horizon', 2)
        self.model_cache_size = getattr(opts, 'vram_orchestrator_cache_size', 3)
    
    def _start_monitoring(self):
        """Start background VRAM monitoring"""
        def monitor_loop():
            while True:
                try:
                    self.vram_monitor.update()
                    self._manage_cache()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    print(f"[VRAM Orchestrator] Monitor error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=monitor_loop, daemon=True, name="vram_orchestrator_monitor")
        thread.start()
    
    def predict_and_preload(self, p: StableDiffusionProcessing):
        """Predict resources and preload them"""
        if not self.enabled or not self.preload_enabled:
            return
        
        try:
            # Generate unique ID for this prediction
            prompt_hash = hashlib.md5(f"{p.prompt}{p.negative_prompt}".encode()).hexdigest()[:16]
            
            # Cancel any pending prediction for same prompt
            if prompt_hash in self.pending_predictions:
                self.pending_predictions[prompt_hash].cancel()
            
            # Submit prediction task
            future = self.preload_executor.submit(
                self._do_prediction_and_preload,
                p, prompt_hash
            )
            self.pending_predictions[prompt_hash] = future
            
            # Clean up old predictions
            self._cleanup_old_predictions()
            
        except Exception as e:
            print(f"[VRAM Orchestrator] Prediction failed: {e}")
    
    def _do_prediction_and_preload(self, p: StableDiffusionProcessing, prompt_hash: str):
        """Execute prediction and preloading"""
        try:
            # Get prediction
            prediction = self.prediction_model.predict(p, self.aggressiveness)
            
            # Check VRAM availability
            vram_status = self.vram_monitor.get_status()
            
            if vram_status['usage_percent'] > self.unload_threshold * 100:
                # High VRAM usage, be conservative
                self._free_vram_for_prediction(prediction)
            
            # Preload resources
            if prediction.checkpoint and prediction.checkpoint != sd_models.model_data.sd_model:
                self._preload_checkpoint(prediction.checkpoint)
            
            if prediction.vae:
                self._preload_vae(prediction.vae)
            
            for lora_name in prediction.loras:
                self._preload_lora(lora_name)
            
            if prediction.upscaler:
                self._preload_upscaler(prediction.upscaler)
            
            # Store prediction in cache
            self.resource_cache[prompt_hash] = {
                'prediction': prediction,
                'timestamp': time.time(),
                'preloaded': True
            }
            
            print(f"[VRAM Orchestrator] Preloaded resources for prompt {prompt_hash[:8]}")
            
        except Exception as e:
            print(f"[VRAM Orchestrator] Preload failed: {e}")
        finally:
            # Remove from pending
            if prompt_hash in self.pending_predictions:
                del self.pending_predictions[prompt_hash]
    
    def _free_vram_for_prediction(self, prediction: ResourcePrediction):
        """Free VRAM by unloading unused resources"""
        # Unload unused checkpoints
        current_checkpoint = sd_models.model_data.sd_model
        if prediction.checkpoint != current_checkpoint:
            # Can unload current if not needed
            pass  # Actual unloading would happen in _preload_checkpoint
        
        # Unload unused LoRAs
        try:
            from modules import extra_networks
            active_loras = extra_networks.active_extra_networks.get('lora', [])
            for lora in active_loras:
                if lora not in prediction.loras:
                    extra_networks.deactivate('lora', [lora])
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _preload_checkpoint(self, checkpoint_name: str):
        """Preload a checkpoint"""
        try:
            # Find checkpoint info
            checkpoint_info = None
            for info in sd_models.checkpoints_list.values():
                if info.name == checkpoint_name:
                    checkpoint_info = info
                    break
            
            if checkpoint_info and checkpoint_info != sd_models.model_data.sd_model:
                # Load checkpoint in background
                sd_models.load_model(checkpoint_info)
                print(f"[VRAM Orchestrator] Preloaded checkpoint: {checkpoint_name}")
        except Exception as e:
            print(f"[VRAM Orchestrator] Failed to preload checkpoint {checkpoint_name}: {e}")
    
    def _preload_vae(self, vae_name: str):
        """Preload a VAE"""
        try:
            vae_path = Path(models_path) / "VAE" / vae_name
            if vae_path.exists():
                sd_vae.load_vae(vae_path)
                print(f"[VRAM Orchestrator] Preloaded VAE: {vae_name}")
        except Exception as e:
            print(f"[VRAM Orchestrator] Failed to preload VAE {vae_name}: {e}")
    
    def _preload_lora(self, lora_name: str):
        """Preload a LoRA"""
        try:
            from modules import extra_networks
            extra_networks.activate('lora', [lora_name])
            print(f"[VRAM Orchestrator] Preloaded LoRA: {lora_name}")
        except Exception as e:
            print(f"[VRAM Orchestrator] Failed to preload LoRA {lora_name}: {e}")
    
    def _preload_upscaler(self, upscaler_name: str):
        """Preload an upscaler model"""
        # Upscalers are typically loaded on-demand
        # We could preload the model file into memory cache
        upscaler_path = Path(models_path) / "ESRGAN" / upscaler_name
        if not upscaler_path.exists():
            upscaler_path = Path(models_path) / "SwinIR" / upscaler_name
        
        if upscaler_path.exists():
            # Read file to warm filesystem cache
            with open(upscaler_path, 'rb') as f:
                f.read(1024)  # Read first 1KB to trigger cache
            print(f"[VRAM Orchestrator] Warmed cache for upscaler: {upscaler_name}")
    
    def _manage_cache(self):
        """Manage resource cache size"""
        if len(self.resource_cache) > self.model_cache_size:
            # Remove oldest entries
            sorted_cache = sorted(
                self.resource_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Keep only newest entries
            for key, _ in sorted_cache[:-self.model_cache_size]:
                del self.resource_cache[key]
    
    def _cleanup_old_predictions(self):
        """Clean up old prediction futures"""
        current_time = time.time()
        to_remove = []
        
        for prompt_hash, future in self.pending_predictions.items():
            if future.done() or (current_time - getattr(future, 'start_time', current_time)) > 30:
                to_remove.append(prompt_hash)
        
        for key in to_remove:
            del self.pending_predictions[key]
    
    def record_usage(self, p: StableDiffusionProcessing, actual_usage: Dict[str, Any]):
        """Record actual resource usage for model training"""
        try:
            features = self.prediction_model.feature_extractor.extract_features(p)
            
            usage = ResourceUsage(
                checkpoint=actual_usage.get('checkpoint', ''),
                vae=actual_usage.get('vae', ''),
                loras=actual_usage.get('loras', []),
                textual_inversions=actual_usage.get('textual_inversions', []),
                upscaler=actual_usage.get('upscaler'),
                adetailer_models=actual_usage.get('adetailer_models', []),
                peak_vram_mb=actual_usage.get('peak_vram_mb', 0),
                generation_time=actual_usage.get('generation_time', 0),
                prompt_hash=hashlib.md5(f"{p.prompt}{p.negative_prompt}".encode()).hexdigest()[:16],
                timestamp=time.time()
            )
            
            self.prediction_model.update(features, usage)
            
        except Exception as e:
            print(f"[VRAM Orchestrator] Failed to record usage: {e}")
    
    def get_prediction_for_prompt(self, prompt: str, negative_prompt: str = "") -> Optional[ResourcePrediction]:
        """Get cached prediction for a prompt"""
        prompt_hash = hashlib.md5(f"{prompt}{negative_prompt}".encode()).hexdigest()[:16]
        cached = self.resource_cache.get(prompt_hash)
        
        if cached and (time.time() - cached['timestamp']) < 300:  # 5 minute cache
            return cached['prediction']
        
        return None
    
    def update_settings(self, **kwargs):
        """Update orchestrator settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save to opts
        opts.vram_orchestrator_enabled = self.enabled
        opts.vram_orchestrator_aggressiveness = self.aggressiveness
        opts.vram_orchestrator_preload = self.preload_enabled
        opts.vram_orchestrator_unload_threshold = self.unload_threshold
        opts.vram_orchestrator_horizon = self.prediction_horizon
        opts.vram_orchestrator_cache_size = self.model_cache_size


class VRAMMonitor:
    """Monitor VRAM usage"""
    
    def __init__(self):
        self.history = []
        self.max_history = 100
        self.last_update = 0
        self.update_interval = 1  # seconds
        
    def update(self):
        """Update VRAM status"""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        try:
            if torch.cuda.is_available():
                # Get VRAM info
                vram_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
                vram_free = vram_total - vram_used
                
                status = {
                    'timestamp': current_time,
                    'used_mb': vram_used,
                    'total_mb': vram_total,
                    'free_mb': vram_free,
                    'usage_percent': (vram_used / vram_total) * 100,
                    'device': torch.cuda.get_device_name(0)
                }
                
                self.history.append(status)
                
                # Trim history
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                    
        except Exception as e:
            print(f"[VRAM Monitor] Failed to update: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current VRAM status"""
        if self.history:
            return self.history[-1]
        
        # Return default if no data
        return {
            'timestamp': time.time(),
            'used_mb': 0,
            'total_mb': 0,
            'free_mb': 0,
            'usage_percent': 0,
            'device': 'unknown'
        }
    
    def get_trend(self, window_seconds: int = 30) -> float:
        """Get VRAM usage trend (positive = increasing)"""
        if len(self.history) < 2:
            return 0.0
        
        current_time = time.time()
        recent = [h for h in self.history if current_time - h['timestamp'] <= window_seconds]
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate linear regression slope
        x = [h['timestamp'] for h in recent]
        y = [h['usage_percent'] for h in recent]
        
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope  # Percentage per second


# Global instance
orchestrator = VRAMOrchestrator()


def on_before_generation(p: StableDiffusionProcessing, *args, **kwargs):
    """Hook called before generation starts"""
    if opts.vram_orchestrator_enabled:
        orchestrator.predict_and_preload(p)


def on_after_generation(p: StableDiffusionProcessing, processed, *args, **kwargs):
    """Hook called after generation completes"""
    if opts.vram_orchestrator_enabled:
        # Record actual usage for model training
        actual_usage = {
            'checkpoint': sd_models.model_data.sd_model.sd_checkpoint_info.name if sd_models.model_data.sd_model else '',
            'vae': sd_vae.loaded_vae_file if hasattr(sd_vae, 'loaded_vae_file') else '',
            'loras': list(extra_networks.active_extra_networks.get('lora', {}).keys()),
            'textual_inversions': [],  # Would need to track this
            'upscaler': getattr(p, 'hr_upscaler', None),
            'adetailer_models': [],  # Would need to track this
            'peak_vram_mb': torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0,
            'generation_time': time.time() - getattr(p, 'start_time', time.time()),
        }
        
        orchestrator.record_usage(p, actual_usage)


def on_ui_settings():
    """Add settings to UI"""
    section = ('vram_orchestrator', "VRAM Orchestrator")
    
    opts.add_option("vram_orchestrator_enabled", shared.OptionInfo(
        True, "Enable VRAM Orchestrator", section=section))
    
    opts.add_option("vram_orchestrator_aggressiveness", shared.OptionInfo(
        0.5, "Prediction Aggressiveness (0.0-1.0)", section=section))
    
    opts.add_option("vram_orchestrator_preload", shared.OptionInfo(
        True, "Enable Preloading", section=section))
    
    opts.add_option("vram_orchestrator_unload_threshold", shared.OptionInfo(
        0.8, "VRAM Unload Threshold (0.0-1.0)", section=section))
    
    opts.add_option("vram_orchestrator_horizon", shared.OptionInfo(
        2, "Prediction Horizon (steps ahead)", section=section))
    
    opts.add_option("vram_orchestrator_cache_size", shared.OptionInfo(
        3, "Model Cache Size", section=section))


# Register callbacks
script_callbacks = None
try:
    from modules import script_callbacks
    script_callbacks.on_before_generation(on_before_generation)
    script_callbacks.on_after_generation(on_after_generation)
    script_callbacks.on_ui_settings(on_ui_settings)
except ImportError:
    print("[VRAM Orchestrator] script_callbacks not available, hooks not registered")


# Integration with existing modules
def patch_lora_loading():
    """Patch LoRA loading to work with orchestrator"""
    try:
        import sys
        lora_module = sys.modules.get('modules.lora.network')
        if lora_module:
            original_load = lora_module.NetworkOnDisk.__init__
            
            def patched_init(self, *args, **kwargs):
                original_load(self, *args, **kwargs)
                # Register with orchestrator
                if hasattr(orchestrator, 'resource_cache'):
                    orchestrator.resource_cache[f"lora_{self.name}"] = {
                        'type': 'lora',
                        'name': self.name,
                        'timestamp': time.time()
                    }
            
            lora_module.NetworkOnDisk.__init__ = patched_init
    except Exception as e:
        print(f"[VRAM Orchestrator] Failed to patch LoRA loading: {e}")


# Initialize patches
patch_lora_loading()


# Export for use in other modules
__all__ = ['orchestrator', 'VRAMOrchestrator', 'ResourcePrediction', 'on_before_generation', 'on_after_generation']