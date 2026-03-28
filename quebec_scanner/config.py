"""
Quebec Business Opportunity Scanner - Configuration
All settings for the autonomous scanner system.
"""

import os

# ─── Telegram Configuration ───
# Set these environment variables or edit directly:
#   export TELEGRAM_BOT_TOKEN="your_token_from_botfather"
#   export TELEGRAM_CHAT_ID="your_chat_id"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ─── Ollama / LLM Configuration ───
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:9b"
LLM_TIMEOUT = 120  # seconds
LLM_MAX_RETRIES = 3
LLM_BATCH_SIZE = 10

# ─── Database ───
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "opportunities.db")

# ─── Scraping Configuration ───
SCRAPE_INTERVAL_HOURS = 6
REQUEST_TIMEOUT = 30
REQUEST_DELAY_MIN = 3.0  # seconds between requests (DDG rate limits at ~1req/2s)
REQUEST_DELAY_MAX = 6.0
MAX_RETRIES = 4
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ─── Quebec Regions & MRCs ───
# Priority MRCs (scanned first and more frequently)
PRIORITY_MRCS = [
    "Brome-Missisquoi",
    "Memphrémagog",
    "Sherbrooke",
    "Coaticook",
    "Le Haut-Saint-François",
    "Le Val-Saint-François",
    "Les Sources",
    "Longueuil",
    "Drummondville",
    "Granby",
]

# All 17 administrative regions of Quebec
REGIONS = {
    "01": {"name": "Bas-Saint-Laurent", "mrcs": ["Kamouraska", "La Matapédia", "La Mitis", "Les Basques", "Rimouski-Neigette", "Rivière-du-Loup", "Témiscouata"]},
    "02": {"name": "Saguenay–Lac-Saint-Jean", "mrcs": ["Le Domaine-du-Roy", "Le Fjord-du-Saguenay", "Lac-Saint-Jean-Est", "Maria-Chapdelaine"]},
    "03": {"name": "Capitale-Nationale", "mrcs": ["Charlevoix", "Charlevoix-Est", "La Côte-de-Beaupré", "La Jacques-Cartier", "L'Île-d'Orléans", "Portneuf", "Québec"]},
    "04": {"name": "Mauricie", "mrcs": ["Les Chenaux", "Maskinongé", "Mékinac", "Shawinigan", "Trois-Rivières"]},
    "05": {"name": "Estrie", "mrcs": ["Brome-Missisquoi", "Coaticook", "Le Granit", "Le Haut-Saint-François", "Le Val-Saint-François", "Les Sources", "Memphrémagog", "Sherbrooke"]},
    "06": {"name": "Montréal", "mrcs": ["Montréal"]},
    "07": {"name": "Outaouais", "mrcs": ["Gatineau", "La Vallée-de-la-Gatineau", "Les Collines-de-l'Outaouais", "Papineau", "Pontiac"]},
    "08": {"name": "Abitibi-Témiscamingue", "mrcs": ["Abitibi", "Abitibi-Ouest", "La Vallée-de-l'Or", "Rouyn-Noranda", "Témiscamingue"]},
    "09": {"name": "Côte-Nord", "mrcs": ["La Haute-Côte-Nord", "Manicouagan", "Minganie", "Sept-Rivières"]},
    "10": {"name": "Nord-du-Québec", "mrcs": ["Baie-James", "Eeyou Istchee"]},
    "11": {"name": "Gaspésie–Îles-de-la-Madeleine", "mrcs": ["Avignon", "Bonaventure", "La Côte-de-Gaspé", "La Haute-Gaspésie", "Les Îles-de-la-Madeleine", "Rocher-Percé"]},
    "12": {"name": "Chaudière-Appalaches", "mrcs": ["Beauce-Centre", "Beauce-Sartigan", "Bellechasse", "L'Islet", "La Nouvelle-Beauce", "Les Appalaches", "Les Etchemins", "Lévis", "Lotbinière", "Montmagny", "Robert-Cliche"]},
    "13": {"name": "Laval", "mrcs": ["Laval"]},
    "14": {"name": "Lanaudière", "mrcs": ["D'Autray", "Joliette", "L'Assomption", "Les Moulins", "Matawinie", "Montcalm"]},
    "15": {"name": "Laurentides", "mrcs": ["Antoine-Labelle", "Argenteuil", "Deux-Montagnes", "La Rivière-du-Nord", "Les Laurentides", "Les Pays-d'en-Haut", "Mirabel", "Thérèse-De Blainville"]},
    "16": {"name": "Montérégie", "mrcs": ["Acton", "Beauharnois-Salaberry", "Brome-Missisquoi", "La Haute-Yamaska", "La Vallée-du-Richelieu", "Le Haut-Richelieu", "Le Haut-Saint-Laurent", "Les Jardins-de-Napierville", "Les Maskoutains", "Longueuil", "Marguerite-D'Youville", "Pierre-De Saurel", "Roussillon", "Rouville", "Vaudreuil-Soulanges"]},
    "17": {"name": "Centre-du-Québec", "mrcs": ["Arthabaska", "Bécancour", "Drummond", "L'Érable", "Nicolet-Yamaska"]},
}

# ─── Business Categories (French keywords for scraping) ───
BUSINESS_CATEGORIES = {
    "construction": {
        "keywords_fr": ["entrepreneur", "rénovation", "construction", "charpentier", "menuisier", "fondation", "agrandissement", "sous-sol"],
        "keywords_en": ["contractor", "renovation", "construction", "carpenter", "framing", "foundation"],
    },
    "plomberie": {
        "keywords_fr": ["plombier", "plomberie", "tuyauterie", "drain", "chauffe-eau", "robinet"],
        "keywords_en": ["plumber", "plumbing", "water heater", "drain"],
    },
    "electricite": {
        "keywords_fr": ["électricien", "électricité", "filage", "panneau électrique", "éclairage"],
        "keywords_en": ["electrician", "electrical", "wiring", "panel"],
    },
    "thermopompe_climatisation": {
        "keywords_fr": ["thermopompe", "climatisation", "air climatisé", "pompe à chaleur", "chauffage", "hvac", "fournaise"],
        "keywords_en": ["heat pump", "air conditioning", "hvac", "furnace", "heating"],
    },
    "toiture": {
        "keywords_fr": ["toiture", "couvreur", "toit", "bardeaux", "gouttière"],
        "keywords_en": ["roofing", "roofer", "shingles", "gutters"],
    },
    "deneigement": {
        "keywords_fr": ["déneigement", "souffleuse", "gratte", "neige", "déblaiement"],
        "keywords_en": ["snow removal", "plowing", "snowblowing"],
    },
    "paysagement": {
        "keywords_fr": ["paysagiste", "paysagement", "aménagement paysager", "gazon", "terrassement", "pavé uni"],
        "keywords_en": ["landscaping", "lawn care", "hardscaping"],
    },
    "menage": {
        "keywords_fr": ["ménage", "nettoyage", "femme de ménage", "entretien ménager", "lavage de vitres"],
        "keywords_en": ["cleaning", "housekeeping", "maid service", "window cleaning"],
    },
    "demenagement": {
        "keywords_fr": ["déménagement", "déménageur", "transport meubles"],
        "keywords_en": ["moving", "movers"],
    },
    "peinture": {
        "keywords_fr": ["peintre", "peinture", "teinture", "décapage"],
        "keywords_en": ["painter", "painting", "staining"],
    },
    "comptabilite": {
        "keywords_fr": ["comptable", "comptabilité", "impôts", "tenue de livres", "déclaration revenus"],
        "keywords_en": ["accountant", "accounting", "bookkeeping", "tax"],
    },
    "traiteur_cuisine": {
        "keywords_fr": ["traiteur", "chef", "cuisine", "repas préparé", "food truck"],
        "keywords_en": ["catering", "chef", "meal prep", "food truck"],
    },
    "sante_bienetre": {
        "keywords_fr": ["physiothérapie", "massothérapie", "ostéopathe", "chiropraticien", "acupuncture"],
        "keywords_en": ["physiotherapy", "massage", "osteopath", "chiropractor"],
    },
    "mecanique_auto": {
        "keywords_fr": ["mécanicien", "mécanique", "garage", "réparation auto", "pneus", "esthétique auto"],
        "keywords_en": ["mechanic", "auto repair", "garage", "tires", "auto detailing"],
    },
    "animaux": {
        "keywords_fr": ["toilettage", "gardiennage animaux", "promenade chien", "vétérinaire", "pension animaux"],
        "keywords_en": ["pet grooming", "pet sitting", "dog walking", "vet"],
    },
    "informatique": {
        "keywords_fr": ["réparation ordinateur", "informatique", "dépannage", "installation réseau", "tech"],
        "keywords_en": ["computer repair", "IT support", "network", "tech support"],
    },
    "evenementiel": {
        "keywords_fr": ["DJ", "photographe", "vidéaste", "décoration événement", "animation"],
        "keywords_en": ["DJ", "photographer", "videographer", "event decoration"],
    },
    "education_tutorat": {
        "keywords_fr": ["tutorat", "tuteur", "cours privé", "aide aux devoirs", "formation"],
        "keywords_en": ["tutoring", "tutor", "private lessons", "homework help"],
    },
    "excavation": {
        "keywords_fr": ["excavation", "terrassement", "drain français", "fosse septique", "nivelage"],
        "keywords_en": ["excavation", "grading", "french drain", "septic"],
    },
    "soudure_metal": {
        "keywords_fr": ["soudure", "soudeur", "métallurgie", "fer forgé", "rampe", "escalier métal"],
        "keywords_en": ["welding", "welder", "metalwork", "wrought iron"],
    },
}

# ─── Kijiji Configuration ───
KIJIJI_BASE_URL = "https://www.kijiji.ca"
KIJIJI_LOCATIONS = {
    "Sherbrooke": "/b-services/sherbrooke/c72l1700150",
    "Granby": "/b-services/granby/c72l1700228",
    "Drummondville": "/b-services/drummondville/c72l1700227",
    "Montreal": "/b-services/ville-de-montreal/c72l1700281",
    "Quebec": "/b-services/ville-de-quebec/c72l1700124",
    "Trois-Rivieres": "/b-services/trois-rivieres/c72l1700191",
    "Gatineau": "/b-services/gatineau/c72l1700245",
    "Saguenay": "/b-services/saguenay/c72l1700232",
    "Rimouski": "/b-services/rimouski/c72l1700141",
    "Laval": "/b-services/laval-rive-nord/c72l1700277",
}

# Kijiji "services wanted" sub-paths
KIJIJI_WANTED_PATHS = [
    "?requestedService=true",  # Filter for service requests
]

# ─── Google Maps / Places Configuration ───
# We use web scraping (no API key needed)
GOOGLE_MAPS_SEARCH_URL = "https://www.google.com/maps/search/"

# ─── REQ (Registraire des entreprises du Québec) ───
REQ_SEARCH_URL = "https://www.registreentreprises.gouv.qc.ca/RQAnonyique/GR/GR03/GR03A2_19A_PIU_RechsijEnt_PC/PageRech.aspx"

# ─── Government Programs ───
GOVERNMENT_PROGRAMS = [
    {
        "name": "Chauffez vert",
        "url": "https://transitionenergetique.gouv.qc.ca/residentiel/programmes/chauffez-vert",
        "sector": "thermopompe_climatisation",
        "description": "Aide financière pour remplacement de systèmes de chauffage",
    },
    {
        "name": "Rénovert",
        "url": "https://www.revenuquebec.ca/fr/citoyens/credits-dimpot/credit-dimpot-renovert/",
        "sector": "construction",
        "description": "Crédit d'impôt pour rénovation écoresponsable",
    },
    {
        "name": "Novoclimat",
        "url": "https://transitionenergetique.gouv.qc.ca/residentiel/programmes/novoclimat",
        "sector": "construction",
        "description": "Programme de construction de maisons écoénergétiques",
    },
    {
        "name": "Logis vert",
        "url": "https://transitionenergetique.gouv.qc.ca/residentiel/programmes",
        "sector": "construction",
        "description": "Programme efficacité énergétique résidentielle",
    },
]

# ─── Scoring Configuration ───
SCORING_WEIGHTS = {
    "demand": 0.30,
    "supply": 0.30,
    "regulatory": 0.20,
    "temporal": 0.20,
}

# Minimum score to trigger a Telegram alert
ALERT_THRESHOLD = 7.0
HIGH_CONFIDENCE_THRESHOLD = 8.0

# ─── Convergence Detection ───
# Minimum number of converging signals for high-confidence opportunity
MIN_CONVERGENCE_SIGNALS = 3

# Thresholds for individual signals
DEMAND_HIGH_THRESHOLD = 15  # requests/month
SUPPLY_LOW_THRESHOLD = 5    # active providers
GROWTH_THRESHOLD = 0.20     # 20% increase
LABOR_SHORTAGE_THRESHOLD = 0.50  # 50% shortage rate

# ─── Logging ───
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "scanner.log")
