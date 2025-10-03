from pathlib import Path
from typing import Annotated
import pandas as pd
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pygal.style import DefaultStyle
from pygal import Bar, Line

customStyle = DefaultStyle
customStyle.font_family = "MartianMono NF"
customStyle.title_font_size = 25
customStyle.label_font_size = 20


def save_bar(data, output_path, title: str) -> None:
    bar_chart = Bar(
        title=title,
        x_title="N objects",
        y_title="Number of Frames",
        show_legend=False,
        style=customStyle,
    )

    bar_chart.add(
        "Frames", [{"value": count, "label": str(count)} for count in data.tolist()]
    )

    bar_chart.x_labels = data.index.astype(str).tolist()

    bar_chart.render_to_file(output_path)


def save_line() -> None: ...


def group_frames(data: pd.DataFrame) -> pd.DataFrame:
    # assert "num_labels" in data.columns

    clusters = data.groupby("frame")["label"].nunique().reset_index(name="num_labels")
    return clusters


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    output_path: Annotated[
        Path,
        Field(
            title="Output directory",
            validation_alias=AliasChoices("o", "output_dir"),
        ),
    ] = Path("data")
    input_dir: Annotated[
        Path,
        Field(
            title="Input video",
            description="Path to video",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]


if __name__ == "__main__":
    args = Arguments()  # pyright: ignore

    for csv_file in args.input_dir.rglob("*.csv"):
        print(csv_file)
        data = pd.read_csv(csv_file)
        clusters = group_frames(data=data)
    
        cluster_counts = clusters["num_labels"].value_counts().sort_index()

        video_name = csv_file.stem
        print(video_name)
        # save_bar(data=cluster_counts, output_path="", title="")
