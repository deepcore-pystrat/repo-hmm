from dataclasses import dataclass

@dataclass
class DataConfig:
    data_path: str
    asset_name : str = "VHP"
