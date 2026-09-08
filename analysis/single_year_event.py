import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

LOGGER = logging.getLogger(__name__)

DATA_FOLDER = Path(__file__).parent.parent / "data"
OUTPUT_FOLDER = Path(__file__).parent.parent / "img"

UNIQUE_ATHLETES_IDENTIFIERS = [
    "Fullname",
    "Sex",
    "BirthYear",
    "CategoryId",
]


def unique_athletes_analysis(df: pl.DataFrame, all_athletes: pl.DataFrame) -> dict:
    # Keep only rows with valid AthleteId values (non-null, non-empty, non-zero)
    individual_df = df.filter(pl.col("AthleteId").is_not_null())
    n_unique_athletes = (
        individual_df.select(UNIQUE_ATHLETES_IDENTIFIERS).unique().shape[0]
    )
    n_unique_male_athletes = (
        individual_df.filter(pl.col("Sex") == "M")
        .select(UNIQUE_ATHLETES_IDENTIFIERS)
        .unique()
        .shape[0]
    )
    n_unique_female_athletes = (
        individual_df.filter(pl.col("Sex") == "F")
        .select(UNIQUE_ATHLETES_IDENTIFIERS)
        .unique()
        .shape[0]
    )

    relay_df = df.filter(pl.col("RelayTeamId").is_not_null())
    n_unique_relayteamids = relay_df.select("RelayTeamId").unique().shape[0]

    # Build the set of unique relay-team-subtitle name fragments
    relay_team_subtitles = (
        relay_df.select("RelayTeamSubtitle").drop_nulls().to_series().to_list()
    )
    relay_team_subtitles_split = [
        subtitle.split("-") for subtitle in relay_team_subtitles
    ]
    relay_team_subtitles_split_flat = [
        item.strip().upper()
        for sublist in relay_team_subtitles_split
        for item in sublist
        if item.strip()  # drop empty fragments from stray "-" characters
    ]
    unique_relay_team_subtitles = set(relay_team_subtitles_split_flat)

    # Corpus of athletes already counted individually — anyone in here is NOT "additional"
    known_fullnames = set(
        individual_df.select("Fullname")
        .drop_nulls()
        .to_series()
        .str.to_uppercase()
        .to_list()
    )

    # Lookup table: normalized Fullname -> Sex, built from the separate all_athletes df
    athlete_sex_lookup = dict(
        all_athletes.drop_nulls(subset=["Fullname", "Sex"])
        .with_columns(pl.col("Fullname").str.strip_chars().str.to_uppercase())
        .select("Fullname", "Sex")
        .unique(subset=["Fullname"], keep="first")
        .iter_rows()
    )

    n_unique_relay_team_subtitle_athletes = -1
    n_additional_male = 0
    n_additional_female = 0
    n_additional_unknown_sex = -1

    for subtitle in unique_relay_team_subtitles:
        if subtitle in known_fullnames:
            continue  # already counted in individual_df

        n_unique_relay_team_subtitle_athletes += 1

        sex = athlete_sex_lookup.get(subtitle)
        if sex == "M":
            n_additional_male += 1
        elif sex == "F":
            n_additional_female += 1
        else:
            n_additional_unknown_sex += 1
            LOGGER.debug(
                f"Could not resolve Sex for relay subtitle athlete: '{subtitle}'"
            )

    n_unique_male_athletes_corrected = n_unique_male_athletes + n_additional_male
    n_unique_female_athletes_corrected = n_unique_female_athletes + n_additional_female
    n_unique_athletes_corrected = (
        n_unique_athletes + n_unique_relay_team_subtitle_athletes
    )

    LOGGER.info(f"Number of unique athletes: {n_unique_athletes}")
    LOGGER.info(f"Number of unique male athletes: {n_unique_male_athletes}")
    LOGGER.info(f"Number of unique female athletes: {n_unique_female_athletes}")
    LOGGER.info(f"Number of unique relay team IDs: {n_unique_relayteamids}")
    LOGGER.info(
        f"Number of additional unique athletes found only in relay subtitles: {n_unique_relay_team_subtitle_athletes}\n"
    )

    return {
        "n_unique_athletes": n_unique_athletes,
        "n_unique_male_athletes": n_unique_male_athletes,
        "n_unique_female_athletes": n_unique_female_athletes,
        "n_unique_relayteamids": n_unique_relayteamids,
        "n_unique_relay_team_subtitle_athletes": n_unique_relay_team_subtitle_athletes,
        "n_additional_male": n_additional_male,
        "n_additional_female": n_additional_female,
        "n_additional_unknown_sex": n_additional_unknown_sex,
        "n_unique_athletes_corrected": n_unique_athletes_corrected,
        "n_unique_male_athletes_corrected": n_unique_male_athletes_corrected,
        "n_unique_female_athletes_corrected": n_unique_female_athletes_corrected,
    }


# Like gender_analysis_old but using CategoryId instead of BirthYear for binning
def gender_analysis(df: pl.DataFrame) -> dict:
    n_male = df.filter(pl.col("Sex") == "M").shape[0]
    n_female = df.filter(pl.col("Sex") == "F").shape[0]

    # Bin by CategoryId
    category_df = (
        df.drop_nulls(subset=["CategoryId"])
        .group_by("CategoryId")
        .agg(
            [
                pl.len().alias("total"),
                (pl.col("Sex") == "M").sum().alias("male"),
                (pl.col("Sex") == "F").sum().alias("female"),
            ]
        )
        .sort("CategoryId")
    )

    category_counts = {
        row["CategoryId"]: {"M": row["male"], "F": row["female"]}
        for row in category_df.to_dicts()
    }

    return {
        "n_male": n_male,
        "n_female": n_female,
        "category_counts": category_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze a single-year event dataset.")
    parser.add_argument(
        "-e",
        "--event",
        type=str,
        help="Event name (used to locate the input Parquet file)",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        help="Year of the event (used to locate the input Parquet file)",
    )
    args = parser.parse_args()

    EVENT_NAME = "UNKNOWN EVENT"

    if args.event == "yc":
        EVENT_NAME = f"YoungChallenge {args.year}"
    elif args.event == "master":
        EVENT_NAME = f"Circuito Master Regionale {args.year}"

    df = pl.read_parquet(DATA_FOLDER / f"{args.event}_{args.year}.parquet")

    ALL_ATHLETES_DF = (
        df.filter(pl.col("AthleteId").is_not_null())
        .select(UNIQUE_ATHLETES_IDENTIFIERS)
        .unique()
    )

    LOGGER.info(f"Total {EVENT_NAME} Data Analysis:")
    unique_athletes_analysis(df, ALL_ATHLETES_DF)

    data = []
    for stage in sorted(df.select("Stage").unique().to_series().to_list()):
        print(f"Stage: {stage}")
        stage_df = df.filter(pl.col("Stage") == stage)
        data.append(unique_athletes_analysis(stage_df, ALL_ATHLETES_DF))

    # Make a plot of the number of unique athletes per stage
    plt.figure(figsize=(10, 6))
    stages = sorted(df.select("Stage").unique().to_series().to_list())
    n_unique_athletes_per_stage = [d["n_unique_athletes_corrected"] for d in data]
    n_unique_male_athletes_per_stage = [
        d["n_unique_male_athletes_corrected"] for d in data
    ]
    n_unique_female_athletes_per_stage = [
        d["n_unique_female_athletes_corrected"] for d in data
    ]
    n_unique_relayteamids_per_stage = [d["n_unique_relayteamids"] for d in data]

    plt.plot(
        stages, n_unique_athletes_per_stage, label="Total Unique Athletes", marker="o"
    )
    plt.plot(
        stages,
        n_unique_male_athletes_per_stage,
        label="Unique Male Athletes",
        marker="o",
    )
    plt.plot(
        stages,
        n_unique_female_athletes_per_stage,
        label="Unique Female Athletes",
        marker="o",
    )
    plt.plot(
        stages, n_unique_relayteamids_per_stage, label="Unique Relay Teams", marker="o"
    )
    plt.xlabel("Stage")
    # Show only integer ticks on the x-axis
    plt.xticks(range(min(stages), len(stages) + min(stages)), stages)
    plt.ylabel("Count")
    plt.grid(ls="--", alpha=0.5)
    ymax = max(max(n_unique_athletes_per_stage), max(n_unique_relayteamids_per_stage))
    plt.ylim(top=ymax * 1.25)
    plt.legend(loc="upper right")
    plt.title(f"{EVENT_NAME}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FOLDER / f"{args.event}_{args.year}_unique_athletes_per_stage.pdf"
    )
    plt.close()

    data = gender_analysis(ALL_ATHLETES_DF)

    categories = list(data["category_counts"].keys())
    # if "U20" is at the end of the list, move it to the front
    if categories[-1] == "U20":
        categories = ["U20"] + categories[:-1]
    if "R" in categories:
        categories = ["R14", "R", "J", "A"]

    plt.figure(figsize=(10, 6))
    male_counts = [data["category_counts"][decade]["M"] for decade in categories]
    female_counts = [data["category_counts"][decade]["F"] for decade in categories]

    x = np.arange(len(categories))  # numeric positions for each decade
    width = 0.4  # width of each bar

    plt.bar(x - width / 2, male_counts, width=width, label="Male", alpha=0.7)
    plt.bar(x + width / 2, female_counts, width=width, label="Female", alpha=0.7)

    plt.xlabel("Category")
    plt.ylabel("Number of Athletes")
    plt.xticks(x, categories)  # put decade labels back on the ticks
    plt.title(f"{EVENT_NAME}")
    plt.legend(loc="upper left")
    plt.grid(ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{args.event}_{args.year}_category.pdf")

    # Plot also the distribution of the sum of male + female athletes per decade
    total_counts = [male + female for male, female in zip(male_counts, female_counts)]
    plt.figure(figsize=(10, 6))
    plt.bar(x, total_counts, width=width, color="gray", alpha=0.7)
    plt.xlabel("Decade of Birth")
    plt.ylabel("Total Number of Athletes")
    plt.xticks(x, categories)  # put decade labels back on the ticks
    plt.title(f"{EVENT_NAME}")
    plt.grid(ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{args.event}_{args.year}_category_total.pdf")


if __name__ == "__main__":
    main()
