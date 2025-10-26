from datetime import datetime
from rich import print as pprint


def loginfo(msg: str) -> None:
    NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pprint(NOW, msg, sep=" ~ ")
