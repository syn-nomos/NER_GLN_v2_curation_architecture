import yaml
from typing import List, Dict

class AutomatedPipeline:
    """
    The orchestrator for the annotation pipeline.
    Loads plugins (automations) and processes text.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.phases = {
            "phase1": False, # Regex
            "phase2": False, # Memory/Suggestions
        }
        # In the future, we will load actual classes here
        
    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return {}

    def set_automation_status(self, phase: str, status: bool):
        """Enables or disables a specific automation phase."""
        if phase in self.phases:
            self.phases[phase] = status

    def process_sentence(self, text: str) -> List[Dict]:
        """
        Runs the enabled automations on the input text.
        Returns a list of suggested annotations.
        """
        suggestions = []
        
        # Placeholder for Phase 1 Logic
        if self.phases.get("phase1"):
            # Call Phase 1 module (to be implemented)
            pass
            
        # Placeholder for Phase 2 Logic
        if self.phases.get("phase2"):
            # Call Phase 2 module (to be implemented)
            pass
            
        return suggestions
