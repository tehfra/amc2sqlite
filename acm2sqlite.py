#!/usr/bin/env python3

import struct
import sys
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, Session


Base = declarative_base()


@dataclass
class CatalogMoviePicture:
    pic_path: str
    pic_data: Optional[bytes] = None
    extension: str = ""

    @property
    def size(self) -> int:
        return len(self.pic_data) if self.pic_data else 0


@dataclass
class CatalogMovie:
    number: int
    checked: bool
    color_tag: int
    media: str
    media_type: str
    source: str
    date: int
    borrower: str
    date_added: int
    user_rating: int
    rating: int
    original_title: str
    translated_title: str
    director: str
    producer: str
    writer: str
    composer: str
    actors: str
    country: str
    year: int
    length: int
    category: str
    certification: str
    url: str
    description: str
    comments: str
    file_path: str
    video_format: str
    video_bitrate: int
    audio_format: str
    audio_bitrate: int
    resolution: str
    framerate: str
    languages: str
    subtitles: str
    size: str
    disks: int
    picture: Optional[CatalogMoviePicture] = None
    custom_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class CatalogCustomFieldProperties:
    field_tag: str
    field_name: str
    field_type: str
    default_value: str


@dataclass
class AntMovieCatalog:
    version: int
    header: str
    custom_fields_properties: List[CatalogCustomFieldProperties]
    movies: List[CatalogMovie]


class AntMovieCatalogReader:
    def __init__(self, file_path: str, debug: bool = False):
        self.file_path = file_path
        self.catalog_bytes = None
        self.byte_offset = 0
        self.debug = debug
        self.header = ""
        self.version = 0
        # Initialize version and header immediately
        self._initialize_version_and_header()

    def _initialize_version_and_header(self):
        """Initialize version and header by checking for ' AMC_' prefix."""
        try:
            with open(self.file_path, "rb") as f:
                header_data = f.read(100)

            if len(header_data) < 100:
                raise ValueError("File too small to contain valid AMC header")

            header_str = header_data.decode("ascii", errors="replace")

            # Check for AMC signature
            if not header_str.startswith(" AMC_"):
                raise ValueError(
                    f"File {self.file_path} is not a valid AMC database - missing ' AMC_' signature"
                )

            # Extract version from header (e.g., " AMC_4.2" -> 42)
            version = 10  # default
            try:
                # Find version pattern like "AMC_4.2"
                amc_pos = header_str.find(" AMC_")
                if amc_pos >= 0:
                    version_part = header_str[amc_pos + 5 : amc_pos + 10]  # Extract "4.2 " part
                    if "." in version_part:
                        major, minor = version_part.split(".", 1)
                        major = major.strip()
                        minor = minor[0] if minor else "0"  # Take first digit of minor
                        if major.isdigit() and minor.isdigit():
                            version = int(major) * 10 + int(minor)
            except (ValueError, IndexError):
                pass  # Keep default version

            self.version = version
            # Get clean header string
            null_pos = header_str.find("\x00")
            self.header = header_str[:null_pos] if null_pos > 0 else header_str[:64]
            self.header = self.header.strip()

            if self.debug:
                print(f"Detected AMC version: {version} from header: '{self.header}'")

        except Exception as e:
            raise ValueError(f"Failed to initialize AMC database {self.file_path}: {e}")

    def load_catalog_file(self):
        with open(self.file_path, "rb") as f:
            self.catalog_bytes = f.read()
        if self.debug:
            print(f"Loaded {len(self.catalog_bytes)} bytes from {self.file_path}")
        # Header and version are already initialized in constructor
        # Set byte offset to start reading data after header
        self.byte_offset = 65

    def _ensure_loaded(self):
        if self.catalog_bytes is None:
            self.load_catalog_file()
        if self.catalog_bytes is None:
            raise ValueError(f"AMC file '{self.file_path}' could not be loaded or is empty.")

    def read_int32_le(self, field_name: str = "") -> int:
        self._ensure_loaded()
        assert self.catalog_bytes is not None
        if self.byte_offset + 4 > len(self.catalog_bytes):
            raise ValueError(
                f"Cannot read i32 for {field_name} at position {self.byte_offset}: end of file"
            )
        value = struct.unpack("<i", self.catalog_bytes[self.byte_offset : self.byte_offset + 4])[0]
        if self.debug and field_name:
            print(f"    {field_name} (pos {self.byte_offset}): {value}")
        self.byte_offset += 4
        return value

    def read_bool_byte(self, field_name: str = "") -> bool:
        self._ensure_loaded()
        assert self.catalog_bytes is not None
        if self.byte_offset + 1 > len(self.catalog_bytes):
            raise ValueError(
                f"Cannot read bool for {field_name} at position {self.byte_offset}: end of file"
            )
        value = self.catalog_bytes[self.byte_offset] != 0
        if self.debug and field_name:
            print(f"    {field_name} (pos {self.byte_offset}): {value}")
        self.byte_offset += 1
        return value

    def read_length_prefixed_string(self, field_name: str = "") -> str:
        self._ensure_loaded()
        assert self.catalog_bytes is not None
        start_pos = self.byte_offset
        length = self.read_int32_le(f"{field_name}_length" if field_name else "string_length")
        if length < 0:
            raise ValueError(
                f"Invalid string length for {field_name}: {length} at position {start_pos}"
            )
        if length == 0:
            if self.debug and field_name:
                print(f"    {field_name} (pos {start_pos}): <empty>")
            return ""
        if length > 100_000:
            raise ValueError(
                f"String too large for {field_name}: {length} bytes at position {start_pos}"
            )
        if self.byte_offset + length > len(self.catalog_bytes):
            raise ValueError(
                f"Cannot read string {field_name} of length {length} at position {self.byte_offset}: end of file"
            )
        string_data = self.catalog_bytes[self.byte_offset : self.byte_offset + length]
        self.byte_offset += length
        text = string_data.decode("latin1", errors="replace")
        printable_count = sum(1 for c in text if c.isprintable() or c in "\t\n\r")
        if len(text) > 20 and printable_count < len(text) * 0.7:
            raise ValueError(f"Data appears to be binary for {field_name} at position {start_pos}")
        if self.debug and field_name:
            preview = text[:50] + "..." if len(text) > 50 else text
            print(f'    {field_name} (pos {start_pos}): "{preview}"')
        return text

    def read_raw_bytes(self, length: int, field_name: str = "") -> bytes:
        self._ensure_loaded()
        assert self.catalog_bytes is not None
        if self.byte_offset + length > len(self.catalog_bytes):
            raise ValueError(
                f"Cannot read {length} bytes for {field_name} at position {self.byte_offset}: end of file"
            )
        data = self.catalog_bytes[self.byte_offset : self.byte_offset + length]
        self.byte_offset += length
        if self.debug and field_name:
            print(f"    {field_name} (pos {self.byte_offset - length}): {length} bytes")
        return data

    def skip_raw_bytes(self, length: int, reason: str = ""):
        self._ensure_loaded()
        assert self.catalog_bytes is not None
        if self.byte_offset + length > len(self.catalog_bytes):
            raise ValueError(
                f"Cannot skip {length} bytes at position {self.byte_offset}: end of file"
            )
        self.byte_offset += length
        if self.debug and reason:
            print(f"    Skipped {length} bytes: {reason} (now at pos {self.byte_offset})")

    def read_custom_field_definitions(self, version: int) -> List[CatalogCustomFieldProperties]:
        self._ensure_loaded()
        if self.debug:
            print("\n=== Reading Custom Fields Properties ===")
        count = self.read_int32_le("custom_fields_count")
        custom_fields = []
        for i in range(count):
            if self.debug:
                print(f"  Reading custom field {i + 1}/{count}")
            field_tag = self.read_length_prefixed_string(f"field_tag_{i}")
            field_name = self.read_length_prefixed_string(f"field_name_{i}")
            field_type = self.read_length_prefixed_string(f"field_type_{i}")
            default_value = self.read_length_prefixed_string(f"default_value_{i}")
            list_values_count = self.read_int32_le(f"list_values_count_{i}")
            for _ in range(list_values_count):
                self.read_length_prefixed_string(f"list_value_{i}_")
            for flag in [
                "list_auto_add",
                "list_sort",
                "list_auto_complete",
                "list_use_catalog_values",
                "multi_values",
                "multi_values_rmp",
                "multi_values_patch",
                "excluded_in_scripts",
            ]:
                self.read_bool_byte(f"{flag}_{i}")
            self.read_raw_bytes(1, f"multi_values_sep_{i}")
            self.read_length_prefixed_string(f"media_info_{i}")
            self.read_length_prefixed_string(f"gui_properties_{i}")
            self.read_length_prefixed_string(f"other_properties_{i}")
            custom_fields.append(
                CatalogCustomFieldProperties(field_tag, field_name, field_type, default_value)
            )
        return custom_fields

    def read_movie_custom_field_values(self) -> Dict[str, str]:
        self._ensure_loaded()
        custom_fields_count = self.read_int32_le("custom_fields_count")
        return {
            self.read_length_prefixed_string(
                f"custom_field_{i}_name"
            ): self.read_length_prefixed_string(f"custom_field_{i}_value")
            for i in range(custom_fields_count)
        }


def parse_hex_pattern_string(pattern_str):
    tokens = pattern_str.strip().split()
    return [(i, bytes.fromhex(token)) for i, token in enumerate(tokens) if token != ".."]


PATTERN_STRING = ".. .. 00 00 .. .. 00 00 .. .. 00 00 .. .. .. .. .. 00 00 00 .. .. .. .. .. .. 00 00 .. .. .. .. .. .. .. .. .. .. .. .. 00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 .. 00 00 00"
MUTUAL_PATTERN = parse_hex_pattern_string(PATTERN_STRING)


def find_movie_pattern_positions(catalog_bytes, scan_byte_limit=None):
    from tqdm import tqdm

    pattern_positions = []
    missing_ids = set()
    last_movie_id = None
    min_offset = min(start for start, _ in MUTUAL_PATTERN)
    max_offset = max(start for start, _ in MUTUAL_PATTERN)
    max_length = max(len(val) for start, val in MUTUAL_PATTERN if start == max_offset)
    window_size = (max_offset + max_length) - min_offset
    scan_byte_limit = (
        min(len(catalog_bytes), scan_byte_limit)
        if scan_byte_limit is not None
        else len(catalog_bytes)
    )
    pattern_len = len(PATTERN_STRING.split())
    pos = 0
    total_steps = scan_byte_limit - window_size + 1
    mb_total = total_steps / 1024 / 1024
    with tqdm(
        total=mb_total,
        desc="Finding movies",
        unit="MB",
        mininterval=0.2,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} {unit} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        while pos < scan_byte_limit - window_size + 1:
            for start, val in MUTUAL_PATTERN:
                if catalog_bytes[pos + start : pos + start + len(val)] != val:
                    break
            else:
                movie_num = (
                    int.from_bytes(catalog_bytes[pos : pos + 2], "little")
                    if window_size >= 2
                    else None
                )
                byte_61 = catalog_bytes[pos + 61] if pos + 61 < len(catalog_bytes) else None
                pattern_positions.append((pos, movie_num, byte_61))
                if (
                    last_movie_id is not None
                    and movie_num is not None
                    and movie_num != last_movie_id + 1
                ):
                    for missing_id in range(last_movie_id + 1, movie_num):
                        missing_ids.add(missing_id)
                if movie_num is not None:
                    last_movie_id = movie_num
                pbar.update(pattern_len / 1024 / 1024)
                pos += pattern_len
                continue
            pbar.update(1 / 1024 / 1024)
            pos += 1
    if missing_ids:
        print(f"Missing (non-consecutive) movie ids: {sorted(missing_ids)}")
    return pattern_positions


def delphi_date_to_datetime(delphi_days):
    """Convert Delphi TDateTime (days since 1899-12-30) to Python datetime."""
    try:
        base_date = datetime(1899, 12, 30)
        return base_date + timedelta(days=delphi_days)
    except (ValueError, OverflowError):
        return None


class PatternBasedAntMovieCatalogReader(AntMovieCatalogReader):
    MOVIE_FIELDS = [
        ("translated_title", "read_length_prefixed_string"),
        ("director", "read_length_prefixed_string"),
        ("producer", "read_length_prefixed_string"),
        ("writer", "read_length_prefixed_string"),
        ("composer", "read_length_prefixed_string"),
        ("country", "read_length_prefixed_string"),
        ("category", "read_length_prefixed_string"),
        ("certification", "read_length_prefixed_string"),
        ("actors", "read_length_prefixed_string"),
        ("url", "read_length_prefixed_string"),
        ("description", "read_length_prefixed_string"),
        ("comments", "read_length_prefixed_string"),
        ("file_path", "read_length_prefixed_string"),
        ("video_format", "read_length_prefixed_string"),
        ("audio_format", "read_length_prefixed_string"),
        ("resolution", "read_length_prefixed_string"),
        ("framerate", "read_length_prefixed_string"),
        ("languages", "read_length_prefixed_string"),
        ("subtitles", "read_length_prefixed_string"),
        ("filesize", "read_length_prefixed_string"),
    ]

    def read_movie_at_pattern_position(
        self,
        pattern_byte_offset: int,
        movie_num: int,
        length: int,
        year: int = 0,
        movie_length: int = 0,
        date_added_raw: int = 0,
        rating_raw: int = 0,
    ) -> Optional[CatalogMovie]:
        movie_data_start = pattern_byte_offset + 65
        if self.debug:
            print(
                f"Reading movie {movie_num} from corrected pattern {pattern_byte_offset:,} + 65 = {movie_data_start:,}"
            )
        self.byte_offset = movie_data_start
        try:
            original_title = (
                self.read_raw_bytes(length, "original_title_data").decode(
                    "latin1", errors="replace"
                )
                if length > 0
                else ""
            )
            field_values = {}
            for field_name, method_name in self.MOVIE_FIELDS:
                value = getattr(self, method_name)(field_name)
                field_values[field_name] = value
            movie_picture = self.read_embedded_movie_picture()
            movie_custom_fields = self.read_movie_custom_field_values()
            if not original_title and field_values["translated_title"]:
                original_title = field_values["translated_title"]

            # Convert Delphi date to readable format for debugging
            date_added_datetime = delphi_date_to_datetime(date_added_raw)
            if self.debug and date_added_datetime:
                print(
                    f"    Date added: {date_added_datetime.strftime('%Y-%m-%d')} (raw: {date_added_raw})"
                )

            # Convert rating for debugging
            if self.debug and rating_raw > 0:
                rating_float = rating_raw / 10.0
                print(f"    Rating: {rating_float:.1f} (raw: {rating_raw})")

            return CatalogMovie(
                number=movie_num,
                checked=False,
                color_tag=0,
                media=f"movie_{movie_num}",
                media_type="",
                source="",
                date=0,
                borrower="",
                date_added=date_added_raw,  # Store raw Delphi date value
                user_rating=0,
                rating=rating_raw,  # Store raw rating value (rating * 10)
                original_title=original_title,
                translated_title=field_values["translated_title"],
                director=field_values["director"],
                producer=field_values["producer"],
                writer=field_values["writer"],
                composer=field_values["composer"],
                actors=field_values["actors"],
                country=field_values["country"],
                year=year,
                length=movie_length,
                category=field_values["category"],
                certification=field_values["certification"],
                url=field_values["url"],
                description=field_values["description"],
                comments=field_values["comments"],
                file_path="",
                video_format="",
                video_bitrate=0,
                audio_format="",
                audio_bitrate=0,
                resolution="",
                framerate="",
                languages=field_values["languages"],
                subtitles="",
                size="",
                disks=0,
                picture=movie_picture,
                custom_fields=movie_custom_fields,
            )
        except Exception as e:
            if self.debug:
                print(f"Error reading movie {movie_num}: {e}")
            return None

    def read_embedded_movie_picture(self) -> Optional[CatalogMoviePicture]:
        extension_length = self.read_int32_le("extension_length")
        if not (0 < extension_length <= 10):
            if self.debug:
                print(f"    No picture (extension_length: {extension_length})")
            return None
        extension = self.read_raw_bytes(extension_length, "extension_data").decode(
            "latin1", errors="replace"
        )
        pic_size = self.read_int32_le("pic_size")
        if pic_size and 1000 < pic_size < 10_000_000:
            if self.debug:
                print(f"    Reading {pic_size} bytes of picture data ({extension})")
            pic_data = self.read_raw_bytes(pic_size, "pic_data")
            return CatalogMoviePicture(f"embedded{extension}", pic_data, extension)
        elif pic_size > 0:
            if self.debug:
                print(f"    Picture size too large ({pic_size}), skipping")
            self.skip_raw_bytes(pic_size, "large picture data")
        return None

    def read_all_movies_by_pattern(self, version: int, scan_byte_limit=None) -> List[CatalogMovie]:
        self.load_catalog_file()
        if self.catalog_bytes is None or len(self.catalog_bytes) == 0:
            raise ValueError(f"AMC file '{self.file_path}' could not be loaded or is empty.")
        pattern_positions = find_movie_pattern_positions(
            self.catalog_bytes, scan_byte_limit=scan_byte_limit
        )
        if pattern_positions:
            highest_movie_id = max(m[1] for m in pattern_positions if m[1] is not None)
            unparsed_movies = highest_movie_id - len(pattern_positions)
            if unparsed_movies == 0:
                print(f"Found {highest_movie_id} movies")
            else:
                print(f"Found {len(pattern_positions)} movies")
                print(f"{unparsed_movies} movies couldn't be parsed")
        from tqdm import tqdm

        catalog_movies = []
        for i, (pattern_byte_offset, movie_num, length) in enumerate(
            tqdm(pattern_positions, desc="Parsing movies", unit="movie")
        ):
            year = struct.unpack(
                "<H",
                self.catalog_bytes[pattern_byte_offset + 20 : pattern_byte_offset + 22],
            )[0]
            movie_length = struct.unpack(
                "<H",
                self.catalog_bytes[pattern_byte_offset + 24 : pattern_byte_offset + 26],
            )[0]

            # Extract date_added from offset +4 (Delphi TDateTime format)
            date_added_raw = struct.unpack(
                "<I",
                self.catalog_bytes[pattern_byte_offset + 4 : pattern_byte_offset + 8],
            )[0]

            # Extract rating from offset +16 (IMDB rating * 10)
            rating_raw = (
                self.catalog_bytes[pattern_byte_offset + 16]
                if pattern_byte_offset + 16 < len(self.catalog_bytes)
                else 0
            )

            movie = self.read_movie_at_pattern_position(
                pattern_byte_offset,
                movie_num,
                length,
                year,
                movie_length,
                date_added_raw,
                rating_raw,
            )
            if movie:
                catalog_movies.append(movie)
        print(f"Successfully read {len(catalog_movies)}/{len(pattern_positions)} movies")
        return catalog_movies

    def read_full_catalog(self, scan_byte_limit=None) -> AntMovieCatalog:
        self.load_catalog_file()
        # Use header and version already read
        custom_fields_properties = (
            self.read_custom_field_definitions(self.version) if self.version >= 40 else []
        )
        catalog_movies = self.read_all_movies_by_pattern(
            self.version, scan_byte_limit=scan_byte_limit
        )
        return AntMovieCatalog(self.version, self.header, custom_fields_properties, catalog_movies)


def sanitize_filename(title: str) -> str:
    """Sanitize a movie title for use as a filename."""
    return (
        "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip().replace(" ", "_")
    )


def extract_embedded_images(catalog_movies: List[CatalogMovie], output_directory: Path):
    output_directory.mkdir(exist_ok=True)
    extracted_count = 0
    for movie in catalog_movies:
        if movie.picture and movie.picture.pic_data:
            title = movie.original_title or movie.translated_title or f"movie_{movie.number}"
            sanitized_title = sanitize_filename(title)
            image_filename = f"{movie.number:04d}_{sanitized_title}{movie.picture.extension}"
            image_filepath = output_directory / image_filename
            with open(image_filepath, "wb") as f:
                f.write(movie.picture.pic_data)
            print(f"Extracted: {image_filename} ({movie.picture.size:,} bytes)")
            extracted_count += 1
    print(f"\nExtracted {extracted_count} images to {output_directory}/")


class CatalogCustomFieldORM(Base):
    __tablename__ = "catalog_custom_field"
    id = Column(Integer, primary_key=True)
    tag = Column(String)
    name = Column(String)
    type = Column(String)
    default_value = Column(String)


class CatalogMovieORM(Base):
    __tablename__ = "catalog_movie"
    number = Column(Integer, primary_key=True)
    original_title = Column(String)
    translated_title = Column(String)
    director = Column(String)
    producer = Column(String)
    writer = Column(String)
    composer = Column(String)
    actors = Column(String)
    country = Column(String)
    year = Column(Integer)
    length = Column(Integer)
    category = Column(String)
    certification = Column(String)
    url = Column(String)
    description = Column(String)
    comments = Column(String)
    date_added = Column(Integer)
    user_rating = Column(Integer)
    rating = Column(Integer)
    checked = Column(Integer)
    color_tag = Column(Integer)
    media = Column(String)
    media_type = Column(String)
    source = Column(String)
    date = Column(Integer)
    borrower = Column(String)
    file_path = Column(String)
    video_format = Column(String)
    video_bitrate = Column(Integer)
    audio_format = Column(String)
    audio_bitrate = Column(Integer)
    resolution = Column(String)
    framerate = Column(String)
    languages = Column(String)
    subtitles = Column(String)
    size = Column(String)
    disks = Column(Integer)
    picture = Column(LargeBinary)
    movie_custom_fields = relationship("CatalogMovieCustomFieldORM", back_populates="movie")


class CatalogMovieCustomFieldORM(Base):
    __tablename__ = "catalog_movie_custom_field"
    id = Column(Integer, primary_key=True)
    movie_number = Column(Integer, ForeignKey("catalog_movie.number"))
    name = Column(String)
    value = Column(String)
    movie = relationship("CatalogMovieORM", back_populates="movie_custom_fields")


def export_catalog_to_sqlalchemy(catalog: AntMovieCatalog, sqlite_path: str):
    """Export AntMovieCatalog data to SQLite using SQLAlchemy ORM."""
    engine = create_engine(f"sqlite:///{sqlite_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # Custom fields
        for custom_field in catalog.custom_fields_properties:
            session.add(
                CatalogCustomFieldORM(
                    tag=custom_field.field_tag,
                    name=custom_field.field_name,
                    type=custom_field.field_type,
                    default_value=custom_field.default_value,
                )
            )
        # Movies and their custom fields
        for movie in catalog.movies:
            movie_orm = CatalogMovieORM(
                number=movie.number,
                original_title=movie.original_title,
                translated_title=movie.translated_title,
                director=movie.director,
                producer=movie.producer,
                writer=movie.writer,
                composer=movie.composer,
                actors=movie.actors,
                country=movie.country,
                year=movie.year,
                length=movie.length,
                category=movie.category,
                certification=movie.certification,
                url=movie.url,
                description=movie.description,
                comments=movie.comments,
                date_added=movie.date_added,
                user_rating=movie.user_rating,
                rating=movie.rating,
                checked=int(movie.checked),
                color_tag=movie.color_tag,
                media=movie.media,
                media_type=movie.media_type,
                source=movie.source,
                date=movie.date,
                borrower=movie.borrower,
                file_path=movie.file_path,
                video_format=movie.video_format,
                video_bitrate=movie.video_bitrate,
                audio_format=movie.audio_format,
                audio_bitrate=movie.audio_bitrate,
                resolution=movie.resolution,
                framerate=movie.framerate,
                languages=movie.languages,
                subtitles=movie.subtitles,
                size=movie.size,
                disks=movie.disks,
                picture=movie.picture.pic_data
                if movie.picture and movie.picture.pic_data
                else None,
            )
            session.add(movie_orm)
            for field_name, field_value in movie.custom_fields.items():
                session.add(
                    CatalogMovieCustomFieldORM(
                        movie_number=movie.number,
                        name=field_name,
                        value=field_value,
                        movie=movie_orm,
                    )
                )
        session.commit()
    print(f"Exported catalog to SQLite file (SQLAlchemy ORM): {sqlite_path}")


def run_main():
    """Main entry point with error handling."""
    parser = argparse.ArgumentParser(description="Consolidated AMC Reader")
    parser.add_argument("amc_file", help="Path to AMC file")
    parser.add_argument(
        "--extract-images",
        type=str,
        metavar="DIR",
        help="Extract embedded images to directory (required if used)",
        required=False,
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        help="Output SQLite file path (if provided, export to SQLite)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    try:
        reader = PatternBasedAntMovieCatalogReader(args.amc_file, debug=args.debug)
        # Print header and version info
        print(f"Ant Movie Catalog database version detected: {reader.version / 10:.1f}", end="")
        if reader.version == 42:
            print(" -- good, this should work.")
        else:
            print(" -- might not work.")
        catalog = reader.read_full_catalog()
        if catalog.movies:
            print("\nFirst 10 movies:")
            for i, movie in enumerate(catalog.movies[:10], 1):
                title = movie.original_title or movie.translated_title or f"Movie {movie.number}"
                print(f"{i}. {title}")
                if movie.director:
                    print(f"   Director: {movie.director}")
                if movie.country:
                    print(f"   Country: {movie.country}")
                if movie.date_added:
                    date_added_dt = delphi_date_to_datetime(movie.date_added)
                    if date_added_dt:
                        print(f"   Date Added: {date_added_dt.strftime('%Y-%m-%d')}")
                if movie.rating > 0:
                    rating_float = movie.rating / 10.0
                    print(f"   Rating: {rating_float:.1f}/10")
                print()
        if args.extract_images is not None:
            print(f"Extracting images to {args.extract_images} ...")
            extract_embedded_images(catalog.movies, Path(args.extract_images))
        if args.sqlite_path:
            export_catalog_to_sqlalchemy(catalog, args.sqlite_path)
    except Exception as e:
        print(f"Error reading AMC file: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_main()
