from pathlib import Path
from typing import Annotated
import pandas as pd
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pygal.style import DefaultStyle
from pygal import Bar, Line

# Custom style setup
customStyle = DefaultStyle
customStyle.title_font_size = 25
customStyle.label_font_size = 20
customStyle.major_label_font_size = 23
customStyle.stroke_width = 2


def save_bar(data, output_path: Path, title: str) -> None:
    """Create and save a bar chart showing distribution of object counts."""
    bar_chart = Bar(
        title=title,
        x_title="Number of objects",
        y_title="Number of Frames",
        show_legend=False,
        style=customStyle,
    )
    bar_chart.add(
        "Frames", [{"value": count, "label": str(count)} for count in data.tolist()]
    )
    bar_chart.x_labels = data.index.astype(str).tolist()
    bar_chart.render_to_file(str(output_path))
    print(f"  Saved bar chart: {output_path}")


def save_line(clusters: pd.DataFrame, output_path: Path, title: str) -> None:
    """Create and save a line chart showing object count over frames."""
    line_chart = Line(
        title=title,
        x_title="Frame",
        y_title="Number of objects",
        style=customStyle,
        show_dots=False,
        truncate_label=-1,
        x_labels=[
            str(frame) if frame % 1000 == 0 else "" for frame in clusters["frame"]
        ],
        width=1600,
        y_labels=clusters["num_labels"].unique().tolist(),
        show_legend=False,
    )
    line_chart.add("Frames", clusters["num_labels"].tolist())
    line_chart.render_to_file(str(output_path))
    print(f"  Saved line chart: {output_path}")


def save_metric_charts(data: pd.DataFrame, output_dir: Path, video_name: str) -> None:
    """Create and save metric charts (min, max, mean) for various measurements."""
    metrics = [
        "area",
        "eccentricity",
        "extent",
        "major_axis_length",
        "max_feret_diameter",
        "minor_axis_length",
        "perimeter",
        "compactness",
        "solidity",
        "sphericity",
    ]

    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        if metric not in data.columns:
            print(f"  Warning: '{metric}' column not found, skipping...")
            continue

        grouped_stats = (
            data.groupby("frame")[metric].agg(["min", "max", "mean"]).reset_index()
        )
        grouped_stats = grouped_stats.dropna(subset=["min", "max", "mean"])

        if grouped_stats.empty:
            print(f"  Warning: No data for '{metric}', skipping...")
            continue

        line_chart_metric = Line(
            style=customStyle,
            x_title="Frame",
            y_title=metric.capitalize(),
            title=f"{video_name} - {metric.capitalize()} Statistics",
            show_dots=False,
            width=1600,
            truncate_label=-1,
            x_labels=[
                str(frame) if frame % 1000 == 0 else ""
                for frame in grouped_stats["frame"]
            ],
        )
        line_chart_metric.add("Min", grouped_stats["min"].tolist())
        line_chart_metric.add("Max", grouped_stats["max"].tolist())
        line_chart_metric.add("Mean", grouped_stats["mean"].tolist())

        output_path = metric_dir / f"{video_name}_{metric}.svg"
        line_chart_metric.render_to_file(str(output_path))
        print(f"  Saved metric chart: {output_path}")


def group_frames(data: pd.DataFrame) -> pd.DataFrame:
    """Group data by frame and count unique labels per frame."""
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
    ] = Path("graphs")
    
    input_path: Annotated[
        Path,
        Field(
            title="Input directory",
            description="Path to directory containing CSV files",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]
    
    pattern: Annotated[
        str,
        Field(
            title="Pattern",
            description="Pattern for CSV files (e.g., '*.csv', 'objects_*.csv')",
            validation_alias=AliasChoices("p", "pattern"),
        ),
    ] = "*.csv"
    
    recursive: Annotated[
        bool,
        Field(
            title="Recursive search",
            description="Search recursively in subdirectories",
            validation_alias=AliasChoices("r", "recursive"),
        ),
    ] = True


if __name__ == "__main__":
    args = Arguments()  # pyright: ignore

    for csv_file in args.input_dir.rglob("*.csv"):
        print(csv_file)
        data = pd.read_csv(csv_file)
        clusters = group_frames(data=data)

        cluster_counts = clusters["num_labels"].value_counts().sort_index()

    obj_count_dir = args.output_path / "obj_count"
    obj_count_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for CSV files in: {args.input_path}")
    print(f"Pattern: {pattern}")
    print(f"Recursive: {args.recursive}")
    print(f"Output directory: {args.output_path}\n")

    if args.recursive:
        csv_files = list(args.input_path.rglob(pattern))
    else:
        csv_files = list(args.input_path.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{pattern}'!")
        exit(1)

    print(f"Found {len(csv_files)} CSV file(s)\n")

    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        try:
            data = pd.read_csv(csv_file)
            
            if "frame" not in data.columns or "label" not in data.columns:
                print(f"  Error: Missing required columns (frame, label). Skipping...\n")
                continue

            clusters = group_frames(data=data)
            cluster_counts = clusters["num_labels"].value_counts().sort_index()
            video_name = csv_file.stem

            bar_output = obj_count_dir / f"{video_name}_hist.svg"
            save_bar(
                data=cluster_counts,
                output_path=bar_output,
                title=f"{video_name} - Object Count Distribution",
            )

            line_output = obj_count_dir / f"{video_name}_line.svg"
            save_line(
                clusters=clusters,
                output_path=line_output,
                title=f"{video_name} - Objects Over Time",
            )

            save_metric_charts(
                data=data, output_dir=args.output_path, video_name=video_name
            )

            print(f"  Successfully processed {video_name}\n")

        except Exception as e:
            print(f"  Error processing {csv_file}: {e}\n")
            continue

    print("Done!")
