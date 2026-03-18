"""modules/resource_predictor.py"""

import os
import re
import json
import time
import threading
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import torch
import numpy as np
from modules import shared, sd_models, sd_vae, extra_networks, ui_components
from modules.script_callbacks import on_app_started, on_before_ui
from modules.ui_components import InputAccordion
import gradio as gr

class PredictionAggressiveness(Enum):
    CONSERVATIVE = 0
    BALANCED = 1
    AGGRESSIVE = 2
    ULTRA = 3

@dataclass
class ResourceProfile:
    vram_required_mb: float = 0.0
    model_hash: Optional[str] = None
    lora_hashes: List[str] = field(default_factory=list)
    vae_hash: Optional[str] = None
    controlnet_models: List[str] = field(default_factory=list)
    estimated_time_ms: float = 0.0
    confidence: float = 0.0

@dataclass
class PromptFeatures:
    prompt_length: int = 0
    negative_prompt_length: int = 0
    has_lora: bool = False
    lora_count: int = 0
    has_controlnet: bool = False
    has_hires: bool = False
    has_upscale: bool = False
    has_adetailer: bool = False
    resolution: Tuple[int, int] = (512, 512)
    batch_size: int = 1
    steps: int = 20
    sampler: str = "Euler a"
    keywords: Set[str] = field(default_factory=set)
    embedding_hashes: List[str] = field(default_factory=list)
    style_hashes: List[str] = field(default_factory=list)

class PromptAnalyzer:
    """Extracts features from prompts for resource prediction."""
    
    LORA_PATTERN = re.compile(r'<lora:([^:]+):[^>]+>', re.IGNORECASE)
    EMBEDDING_PATTERN = re.compile(r'\bembd?:([^:\s]+)', re.IGNORECASE)
    STYLE_PATTERN = re.compile(r'\bstyle:([^:\s]+)', re.IGNORECASE)
    CONTROLNET_PATTERN = re.compile(r'\bcontrolnet\b', re.IGNORECASE)
    HIRES_KEYWORDS = {'hires', 'upscale', 'highres', 'high resolution', '4k', '8k'}
    UPSCALE_KEYWORDS = {'upscale', 'esrgan', 'realesrgan', 'swinir', '4x', '8x'}
    ADETAILER_KEYWORDS = {'adetailer', 'after detailer', 'face fix', 'face restore'}
    
    @classmethod
    def extract_features(cls, prompt: str, negative_prompt: str, 
                         width: int, height: int, steps: int, 
                         batch_size: int, sampler_name: str) -> PromptFeatures:
        features = PromptFeatures()
        features.prompt_length = len(prompt)
        features.negative_prompt_length = len(negative_prompt)
        features.resolution = (width, height)
        features.batch_size = batch_size
        features.steps = steps
        features.sampler = sampler_name
        
        # Extract LoRA references
        lora_matches = cls.LORA_PATTERN.findall(prompt + negative_prompt)
        features.lora_count = len(lora_matches)
        features.has_lora = features.lora_count > 0
        
        # Extract embedding references
        features.embedding_hashes = cls.EMBEDDING_PATTERN.findall(prompt + negative_prompt)
        
        # Extract style references
        features.style_hashes = cls.STYLE_PATTERN.findall(prompt + negative_prompt)
        
        # Detect ControlNet usage
        features.has_controlnet = bool(cls.CONTROLNET_PATTERN.search(prompt + negative_prompt))
        
        # Detect keyword-based features
        prompt_lower = prompt.lower()
        negative_lower = negative_prompt.lower()
        combined = prompt_lower + " " + negative_lower
        
        features.has_hires = any(keyword in combined for keyword in cls.HIRES_KEYWORDS)
        features.has_upscale = any(keyword in combined for keyword in cls.UPSCALE_KEYWORDS)
        features.has_adetailer = any(keyword in combined for keyword in cls.ADETAILER_KEYWORDS)
        
        # Extract keywords for model prediction
        words = set(re.findall(r'\b[a-z_]{3,}\b', combined))
        features.keywords = words - {'the', 'and', 'with', 'for', 'that', 'this', 'from'}
        
        return features

class ResourcePredictor:
    """AI-driven predictor for resource needs based on prompt analysis."""
    
    MODEL_CACHE_FILE = os.path.join(shared.models_path, "resource_predictor_cache.json")
    HISTORY_FILE = os.path.join(shared.models_path, "resource_history.json")
    
    def __init__(self):
        self.enabled = True
        self.aggressiveness = PredictionAggressiveness.BALANCED
        self.history: List[Dict] = []
        self.model_mapping: Dict[str, ResourceProfile] = {}
        self.keyword_weights: Dict[str, float] = defaultdict(float)
        self.current_profile: Optional[ResourceProfile] = None
        self.preloaded_models: Dict[str, Any] = {}
        self.vram_budget_mb = self._detect_vram()
        self.load_history()
        self.load_model_mapping()
        self._initialize_default_mappings()
        
        # Thread for background preloading
        self.preload_thread = None
        self.stop_preload = threading.Event()
        
        # Statistics
        self.prediction_count = 0
        self.accurate_predictions = 0
        self.vram_saved_mb = 0.0
        self.time_saved_ms = 0.0
        
        # UI components
        self.ui_initialized = False
        
    def _detect_vram(self) -> float:
        """Detect available VRAM in MB."""
        if torch.cuda.is_available():
            try:
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            except:
                pass
        return 8192  # Default assumption
    
    def load_history(self):
        """Load prediction history from disk."""
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading resource history: {e}")
            self.history = []
    
    def save_history(self):
        """Save prediction history to disk."""
        try:
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(self.history[-1000:], f)  # Keep last 1000 entries
        except Exception as e:
            print(f"Error saving resource history: {e}")
    
    def load_model_mapping(self):
        """Load model resource mappings from disk."""
        try:
            if os.path.exists(self.MODEL_CACHE_FILE):
                with open(self.MODEL_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.model_mapping = {
                        k: ResourceProfile(**v) for k, v in data.get('mappings', {}).items()
                    }
                    self.keyword_weights = defaultdict(float, data.get('weights', {}))
        except Exception as e:
            print(f"Error loading model mappings: {e}")
    
    def save_model_mapping(self):
        """Save model resource mappings to disk."""
        try:
            data = {
                'mappings': {
                    k: {
                        'vram_required_mb': v.vram_required_mb,
                        'model_hash': v.model_hash,
                        'lora_hashes': v.lora_hashes,
                        'vae_hash': v.vae_hash,
                        'controlnet_models': v.controlnet_models,
                        'estimated_time_ms': v.estimated_time_ms,
                        'confidence': v.confidence
                    }
                    for k, v in self.model_mapping.items()
                },
                'weights': dict(self.keyword_weights)
            }
            with open(self.MODEL_CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving model mappings: {e}")
    
    def _initialize_default_mappings(self):
        """Initialize default resource profiles for common models."""
        if not self.model_mapping:
            # Default SD 1.5 models
            self.model_mapping.update({
                'sd_1.5': ResourceProfile(vram_required_mb=2048, estimated_time_ms=5000),
                'sd_2.1': ResourceProfile(vram_required_mb=2560, estimated_time_ms=6000),
                'sdxl': ResourceProfile(vram_required_mb=6144, estimated_time_ms=10000),
                'sd_xl_turbo': ResourceProfile(vram_required_mb=4096, estimated_time_ms=3000),
            })
    
    def analyze_prompt(self, prompt: str, negative_prompt: str = "",
                       width: int = 512, height: int = 512,
                       steps: int = 20, batch_size: int = 1,
                       sampler_name: str = "Euler a") -> PromptFeatures:
        """Analyze prompt and extract features for prediction."""
        return PromptAnalyzer.extract_features(
            prompt, negative_prompt, width, height, 
            steps, batch_size, sampler_name
        )
    
    def predict_resource_needs(self, features: PromptFeatures) -> ResourceProfile:
        """Predict resource requirements based on prompt features."""
        profile = ResourceProfile()
        
        # Base VRAM estimation
        base_vram = 1024  # Base model overhead
        
        # Resolution scaling
        pixel_count = features.resolution[0] * features.resolution[1]
        resolution_factor = pixel_count / (512 * 512)
        base_vram *= resolution_factor
        
        # Batch size scaling
        base_vram *= features.batch_size
        
        # Steps impact (minimal on VRAM, affects time)
        time_factor = features.steps / 20
        
        # LoRA overhead
        lora_overhead = features.lora_count * 256  # ~256MB per LoRA
        
        # ControlNet overhead
        controlnet_overhead = 1024 if features.has_controlnet else 0
        
        # Hires/upscale overhead
        hires_overhead = 2048 if features.has_hires else 0
        upscale_overhead = 1024 if features.has_upscale else 0
        
        # ADetailer overhead
        adetailer_overhead = 512 if features.has_adetailer else 0
        
        # Calculate total VRAM
        total_vram = (base_vram + lora_overhead + controlnet_overhead + 
                     hires_overhead + upscale_overhead + adetailer_overhead)
        
        # Apply aggressiveness factor
        aggressiveness_factor = {
            PredictionAggressiveness.CONSERVATIVE: 1.0,
            PredictionAggressiveness.BALANCED: 1.1,
            PredictionAggressiveness.AGGRESSIVE: 1.25,
            PredictionAggressiveness.ULTRA: 1.5
        }[self.aggressiveness]
        
        profile.vram_required_mb = total_vram * aggressiveness_factor
        
        # Predict model based on keywords
        predicted_model = self._predict_model_from_keywords(features.keywords)
        if predicted_model and predicted_model in self.model_mapping:
            profile.model_hash = predicted_model
            profile.vram_required_mb = max(
                profile.vram_required_mb,
                self.model_mapping[predicted_model].vram_required_mb
            )
        
        # Estimate time based on model and steps
        base_time = 5000  # 5 seconds base
        if profile.model_hash and profile.model_hash in self.model_mapping:
            base_time = self.model_mapping[profile.model_hash].estimated_time_ms
        
        profile.estimated_time_ms = base_time * time_factor * features.batch_size
        
        # Calculate confidence based on feature matches
        confidence = 0.5  # Base confidence
        if features.keywords:
            confidence += min(len(features.keywords) * 0.05, 0.3)
        if profile.model_hash:
            confidence += 0.2
        profile.confidence = min(confidence, 1.0)
        
        return profile
    
    def _predict_model_from_keywords(self, keywords: Set[str]) -> Optional[str]:
        """Predict which model will be used based on keywords."""
        if not keywords:
            return None
        
        # Score each model based on keyword weights
        scores = defaultdict(float)
        for keyword in keywords:
            for model_hash, weight in self.keyword_weights.items():
                if keyword in model_hash.lower():
                    scores[model_hash] += weight
        
        # Return highest scoring model if above threshold
        if scores:
            best_model = max(scores.items(), key=lambda x: x[1])
            if best_model[1] > 0.3:  # Minimum confidence threshold
                return best_model[0]
        
        return None
    
    def learn_from_generation(self, prompt: str, negative_prompt: str,
                             model_hash: str, lora_hashes: List[str],
                             vae_hash: Optional[str], vram_used_mb: float,
                             generation_time_ms: float, width: int, height: int,
                             steps: int, batch_size: int, sampler_name: str):
        """Learn from completed generation to improve predictions."""
        features = self.analyze_prompt(
            prompt, negative_prompt, width, height, 
            steps, batch_size, sampler_name
        )
        
        # Update model mapping
        if model_hash not in self.model_mapping:
            self.model_mapping[model_hash] = ResourceProfile()
        
        profile = self.model_mapping[model_hash]
        
        # Update with exponential moving average
        alpha = 0.3  # Learning rate
        profile.vram_required_mb = (alpha * vram_used_mb + 
                                   (1 - alpha) * profile.vram_required_mb)
        profile.estimated_time_ms = (alpha * generation_time_ms + 
                                    (1 - alpha) * profile.estimated_time_ms)
        profile.model_hash = model_hash
        profile.lora_hashes = lora_hashes
        profile.vae_hash = vae_hash
        profile.confidence = min(profile.confidence + 0.1, 1.0)
        
        # Update keyword weights
        for keyword in features.keywords:
            # Increase weight for keywords associated with this model
            self.keyword_weights[model_hash] = (
                self.keyword_weights.get(model_hash, 0) + 0.1
            )
        
        # Save to history
        history_entry = {
            'timestamp': time.time(),
            'prompt_hash': hashlib.md5(prompt.encode()).hexdigest()[:8],
            'model_hash': model_hash,
            'vram_used_mb': vram_used_mb,
            'generation_time_ms': generation_time_ms,
            'features': {
                'resolution': list(features.resolution),
                'steps': features.steps,
                'batch_size': features.batch_size,
                'has_lora': features.has_lora,
                'lora_count': features.lora_count,
                'has_controlnet': features.has_controlnet,
            }
        }
        self.history.append(history_entry)
        
        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Periodically save to disk
        if len(self.history) % 10 == 0:
            self.save_history()
            self.save_model_mapping()
    
    def preload_models(self, profile: ResourceProfile):
        """Preload models based on prediction."""
        if not self.enabled or not profile.model_hash:
            return
        
        def preload_task():
            try:
                # Check if we need to load a new model
                current_model = sd_models.model_data.sd_model
                current_hash = None
                if current_model:
                    current_hash = getattr(current_model, 'hash', None)
                
                # Load main model if different
                if profile.model_hash and profile.model_hash != current_hash:
                    if self._should_preload_model(profile):
                        print(f"ResourcePredictor: Preloading model {profile.model_hash}")
                        sd_models.load_model(profile.model_hash)
                
                # Preload LoRAs
                for lora_hash in profile.lora_hashes:
                    if lora_hash not in self.preloaded_models:
                        # This would integrate with LoRA loading system
                        pass
                
                # Preload VAE if specified
                if profile.vae_hash:
                    # This would integrate with VAE loading system
                    pass
                
                self.preloaded_models[profile.model_hash] = time.time()
                
            except Exception as e:
                print(f"ResourcePredictor: Error during preloading: {e}")
        
        # Start preloading in background thread
        if self.preload_thread and self.preload_thread.is_alive():
            self.stop_preload.set()
            self.preload_thread.join(timeout=1.0)
        
        self.stop_preload.clear()
        self.preload_thread = threading.Thread(target=preload_task, daemon=True)
        self.preload_thread.start()
    
    def _should_preload_model(self, profile: ResourceProfile) -> bool:
        """Determine if we should preload based on aggressiveness and VRAM."""
        available_vram = self.vram_budget_mb - self._get_current_vram_usage()
        
        if available_vram < profile.vram_required_mb:
            return False
        
        # Aggressiveness-based decision
        thresholds = {
            PredictionAggressiveness.CONSERVATIVE: 0.9,  # Only preload if 90% sure
            PredictionAggressiveness.BALANCED: 0.7,
            PredictionAggressiveness.AGGRESSIVE: 0.5,
            PredictionAggressiveness.ULTRA: 0.3
        }
        
        return profile.confidence >= thresholds[self.aggressiveness]
    
    def _get_current_vram_usage(self) -> float:
        """Get current VRAM usage in MB."""
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated(0) / (1024 ** 2)
            except:
                pass
        return 0.0
    
    def preempt_unneeded_models(self, needed_hashes: Set[str]):
        """Unload models that won't be needed."""
        if not self.enabled:
            return
        
        # This would integrate with the model unloading system
        # For now, just clean up our cache
        current_time = time.time()
        to_remove = []
        
        for model_hash, load_time in self.preloaded_models.items():
            if model_hash not in needed_hashes:
                # Keep models loaded for a while based on aggressiveness
                keep_time = {
                    PredictionAggressiveness.CONSERVATIVE: 300,  # 5 minutes
                    PredictionAggressiveness.BALANCED: 180,      # 3 minutes
                    PredictionAggressiveness.AGGRESSIVE: 60,     # 1 minute
                    PredictionAggressiveness.ULTRA: 30           # 30 seconds
                }[self.aggressiveness]
                
                if current_time - load_time > keep_time:
                    to_remove.append(model_hash)
        
        for model_hash in to_remove:
            del self.preloaded_models[model_hash]
    
    def update_aggressiveness(self, value: int):
        """Update prediction aggressiveness setting."""
        self.aggressiveness = PredictionAggressiveness(value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        accuracy = (self.accurate_predictions / self.prediction_count * 100 
                   if self.prediction_count > 0 else 0)
        
        return {
            'enabled': self.enabled,
            'aggressiveness': self.aggressiveness.name,
            'prediction_count': self.prediction_count,
            'accuracy_percent': round(accuracy, 1),
            'vram_saved_mb': round(self.vram_saved_mb, 1),
            'time_saved_ms': round(self.time_saved_ms, 1),
            'vram_budget_mb': round(self.vram_budget_mb, 1),
            'current_vram_usage_mb': round(self._get_current_vram_usage(), 1),
            'models_cached': len(self.preloaded_models),
            'models_mapped': len(self.model_mapping),
            'keywords_learned': len(self.keyword_weights)
        }
    
    def create_ui(self):
        """Create UI components for resource predictor settings."""
        if self.ui_initialized:
            return
        
        with gr.Accordion("Resource Predictor", open=False):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable predictive resource orchestration",
                    value=self.enabled,
                    interactive=True
                )
                
                aggressiveness = gr.Slider(
                    label="Prediction aggressiveness",
                    minimum=0,
                    maximum=3,
                    step=1,
                    value=self.aggressiveness.value,
                    interactive=True
                )
            
            with gr.Row():
                stats_btn = gr.Button("Show Statistics")
                clear_btn = gr.Button("Clear History")
            
            stats_output = gr.JSON(label="Statistics", visible=False)
            
            # Event handlers
            enabled.change(
                fn=lambda x: setattr(self, 'enabled', x),
                inputs=[enabled],
                outputs=[]
            )
            
            aggressiveness.change(
                fn=self.update_aggressiveness,
                inputs=[aggressiveness],
                outputs=[]
            )
            
            stats_btn.click(
                fn=lambda: gr.update(value=self.get_statistics(), visible=True),
                inputs=[],
                outputs=[stats_output]
            )
            
            clear_btn.click(
                fn=lambda: (self.history.clear(), self.save_history(), "History cleared"),
                inputs=[],
                outputs=[]
            )
        
        self.ui_initialized = True

# Global instance
predictor = ResourcePredictor()

# Integration hooks
def on_app_started(demo, app):
    """Initialize predictor when app starts."""
    predictor.create_ui()

def on_before_ui():
    """Called before UI is rendered."""
    pass

# Register callbacks
on_app_started(None, None)
on_before_ui()

# Monkey-patch integration points
_original_process_images = None

def integrate_with_processing():
    """Integrate predictor with image processing pipeline."""
    global _original_process_images
    
    try:
        from modules import processing
        
        _original_process_images = processing.process_images
        
        def enhanced_process_images(p):
            """Enhanced process_images with resource prediction."""
            if not predictor.enabled:
                return _original_process_images(p)
            
            # Extract prompt information
            prompt = p.prompt if hasattr(p, 'prompt') else ""
            negative_prompt = p.negative_prompt if hasattr(p, 'negative_prompt') else ""
            width = p.width if hasattr(p, 'width') else 512
            height = p.height if hasattr(p, 'height') else 512
            steps = p.steps if hasattr(p, 'steps') else 20
            batch_size = p.batch_size if hasattr(p, 'batch_size') else 1
            sampler_name = p.sampler_name if hasattr(p, 'sampler_name') else "Euler a"
            
            # Analyze and predict
            features = predictor.analyze_prompt(
                prompt, negative_prompt, width, height,
                steps, batch_size, sampler_name
            )
            
            profile = predictor.predict_resource_needs(features)
            predictor.current_profile = profile
            predictor.prediction_count += 1
            
            # Preload predicted resources
            predictor.preload_models(profile)
            
            # Record start time for learning
            start_time = time.time()
            
            # Execute original processing
            result = _original_process_images(p)
            
            # Learn from completed generation
            generation_time = (time.time() - start_time) * 1000
            
            # Get actual model information
            model_hash = None
            if hasattr(p, 'sd_model') and p.sd_model:
                model_hash = getattr(p.sd_model, 'hash', None)
            
            lora_hashes = []
            if hasattr(p, 'lora_hashes'):
                lora_hashes = p.lora_hashes
            
            vae_hash = None
            if hasattr(p, 'sd_vae'):
                vae_hash = p.sd_vae
            
            # Estimate VRAM usage (simplified)
            estimated_vram = profile.vram_required_mb
            
            # Learn from this generation
            if model_hash:
                predictor.learn_from_generation(
                    prompt, negative_prompt, model_hash, lora_hashes,
                    vae_hash, estimated_vram, generation_time,
                    width, height, steps, batch_size, sampler_name
                )
            
            # Clean up unneeded models
            needed_hashes = {model_hash} if model_hash else set()
            needed_hashes.update(lora_hashes)
            if vae_hash:
                needed_hashes.add(vae_hash)
            
            predictor.preempt_unneeded_models(needed_hashes)
            
            return result
        
        # Apply monkey patch
        processing.process_images = enhanced_process_images
        print("ResourcePredictor: Integrated with processing pipeline")
        
    except Exception as e:
        print(f"ResourcePredictor: Failed to integrate with processing: {e}")

# Initialize integration
try:
    integrate_with_processing()
except:
    pass

# Export public API
__all__ = ['ResourcePredictor', 'predictor', 'PredictionAggressiveness']