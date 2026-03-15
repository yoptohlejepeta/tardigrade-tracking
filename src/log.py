from datetime import datetime
from typing import Any
from rich import print as pprint


def loginfo(msg: Any) -> None:
    NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pprint(NOW, msg, sep=" ~ ")
