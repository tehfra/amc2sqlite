#!/usr/bin/env python3

import struct
import sys
import argparse
from typing import Optional, List, Dict, BinaryIO
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta

# SQLAlchemy imports for database functionality
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    LargeBinary,
    ForeignKey,
    Text,
    Boolean,
)
from sqlalchemy.orm import declarative_base, relationship, Session

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CatalogMoviePicture:
    pic_path: str
    pic_data: Optional[bytes] = None
    extension: str = ""

    @property
    def size(self) -> int:
        return len(self.pic_data) if self.pic_data else 0


@dataclass
class CatalogMovieExtra:
    checked: bool
    tag: str
    title: str
    category: str
    url: str
    description: str
    comments: str
    created_by: str
    picture: Optional[CatalogMoviePicture] = None


@dataclass
class CatalogMovie:
    number: int
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
    user_rating: int
    rating: int
    date_watched: int
    date_added: int
    checked: bool
    color_tag: int
    media: str
    media_type: str
    source: str
    borrower: str
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
    extras: List[CatalogMovieExtra] = field(default_factory=list)


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


# ============================================================================
# SQLAlchemy Database Schema
# ============================================================================

Base = declarative_base()


class CatalogPropertiesORM(Base):
    """Catalog owner and properties information."""

    __tablename__ = "catalog_properties"

    id = Column(Integer, primary_key=True)
    version = Column(Integer)
    header = Column(String)
    owner_name = Column(String)
    owner_mail = Column(String)
    owner_site = Column(String)
    description = Column(Text)


class CatalogCustomFieldORM(Base):
    """Custom field definitions."""

    __tablename__ = "catalog_custom_field"

    id = Column(Integer, primary_key=True)
    tag = Column(String, unique=True)
    name = Column(String)
    type = Column(String)
    default_value = Column(String)


class CatalogMovieORM(Base):
    """Main movie data."""

    __tablename__ = "catalog_movie"

    number = Column(Integer, primary_key=True)
    original_title = Column(String)
    translated_title = Column(String)
    director = Column(String)
    producer = Column(String)
    writer = Column(String)
    composer = Column(String)
    actors = Column(Text)
    country = Column(String)
    year = Column(Integer)
    length = Column(Integer)
    category = Column(String)
    certification = Column(String)
    url = Column(String)
    description = Column(Text)
    comments = Column(Text)
    user_rating = Column(Integer)
    rating = Column(Integer)
    date_watched = Column(Integer)
    date_added = Column(Integer)
    checked = Column(Boolean)
    color_tag = Column(Integer)
    media = Column(String)
    media_type = Column(String)
    source = Column(String)
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

    # Image data
    picture_path = Column(String)
    picture_data = Column(LargeBinary)
    picture_extension = Column(String)

    # Relationships
    custom_fields = relationship(
        "CatalogMovieCustomFieldORM",
        back_populates="movie",
        cascade="all, delete-orphan",
    )
    extras = relationship(
        "CatalogMovieExtraORM", back_populates="movie", cascade="all, delete-orphan"
    )


class CatalogMovieCustomFieldORM(Base):
    """Custom field values for movies."""

    __tablename__ = "catalog_movie_custom_field"

    id = Column(Integer, primary_key=True)
    movie_number = Column(Integer, ForeignKey("catalog_movie.number"))
    field_tag = Column(String)
    field_value = Column(Text)

    # Relationships
    movie = relationship("CatalogMovieORM", back_populates="custom_fields")


class CatalogMovieExtraORM(Base):
    """Movie extras/supplements."""

    __tablename__ = "catalog_movie_extra"

    id = Column(Integer, primary_key=True)
    movie_number = Column(Integer, ForeignKey("catalog_movie.number"))
    checked = Column(Boolean)
    tag = Column(String)
    title = Column(String)
    category = Column(String)
    url = Column(String)
    description = Column(Text)
    comments = Column(Text)
    created_by = Column(String)

    # Image data
    picture_path = Column(String)
    picture_data = Column(LargeBinary)
    picture_extension = Column(String)

    # Relationships
    movie = relationship("CatalogMovieORM", back_populates="extras")


# ============================================================================
# AMC Parser
# ============================================================================


class AntMovieCatalogReader:
    def __init__(self, file_path: str, debug_level: int = 0, buffer_size: int = 8192):
        self.file_path = file_path
        self.debug_level = debug_level  # 0=none, 1=basic, 2=detailed
        self.buffer_size = buffer_size
        self.file: Optional[BinaryIO] = None
        self.file_size = 0
        self.position = 0
        self.version = 0
        self.header = ""

        # Debug statistics
        self.stats = {"strings_read": 0, "bytes_read": 0, "movies_read": 0}

        # Initialize version and header
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
                amc_pos = header_str.find(" AMC_")
                if amc_pos >= 0:
                    version_part = header_str[amc_pos + 5 : amc_pos + 10]
                    if "." in version_part:
                        major, minor = version_part.split(".", 1)
                        major = major.strip()
                        minor = minor[0] if minor else "0"
                        if major.isdigit() and minor.isdigit():
                            version = int(major) * 10 + int(minor)
            except (ValueError, IndexError):
                pass

            self.version = version
            null_pos = header_str.find("\x00")
            self.header = header_str[:null_pos] if null_pos > 0 else header_str[:64]
            self.header = self.header.strip()

            if self.debug_level >= 1:
                print(f"Detected AMC version: {version} from header: '{self.header}'")

        except Exception as e:
            raise ValueError(f"Failed to initialize AMC database {self.file_path}: {e}")

    def _open_file(self):
        """Open file for streaming if not already open."""
        if self.file is None:
            self.file = open(self.file_path, "rb")
            self.file_size = Path(self.file_path).stat().st_size
            self.position = 65  # Skip header
            self.file.seek(65)

            if self.debug_level >= 1:
                print(f"Opened {self.file_path} ({self.file_size:,} bytes) for streaming")

    def _debug_log(self, field_name: str, value, pos: int):
        """Efficient debug logging."""
        if self.debug_level >= 2 and field_name:
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"    {field_name} (pos {pos}): {value}")

    def read_int32_le(self, field_name: str = "") -> int:
        """Read 32-bit little-endian integer."""
        self._open_file()
        assert self.file is not None  # Type assertion for linter
        data = self.file.read(4)
        if len(data) != 4:
            raise ValueError(
                f"Cannot read i32 for {field_name} at position {self.position}: end of file"
            )

        value = struct.unpack("<i", data)[0]
        self._debug_log(field_name, value, self.position)
        self.position += 4
        self.stats["bytes_read"] += 4
        return value

    def read_bool_byte(self, field_name: str = "") -> bool:
        """Read boolean byte."""
        self._open_file()
        assert self.file is not None  # Type assertion for linter
        data = self.file.read(1)
        if len(data) != 1:
            raise ValueError(
                f"Cannot read bool for {field_name} at position {self.position}: end of file"
            )

        value = data[0] != 0
        self._debug_log(field_name, value, self.position)
        self.position += 1
        self.stats["bytes_read"] += 1
        return value

    def read_length_prefixed_string(self, field_name: str = "", validate: bool = False) -> str:
        """
        Read length-prefixed string with optional validation.
        Validation is disabled by default for performance.
        """
        start_pos = self.position
        length = self.read_int32_le(f"{field_name}_length" if field_name else "string_length")

        if length < 0:
            raise ValueError(
                f"Invalid string length for {field_name}: {length} at position {start_pos}"
            )

        if length == 0:
            if self.debug_level >= 2 and field_name:
                print(f"    {field_name} (pos {start_pos}): <empty>")
            return ""

        # Sanity check for extremely large strings
        if length > 10_000_000:  # 10MB limit
            raise ValueError(
                f"String too large for {field_name}: {length} bytes at position {start_pos}"
            )

        self._open_file()
        assert self.file is not None  # Type assertion for linter
        string_data = self.file.read(length)
        if len(string_data) != length:
            raise ValueError(
                f"Cannot read string {field_name} of length {length} at position {self.position}: end of file"
            )

        self.position += length
        self.stats["bytes_read"] += length
        self.stats["strings_read"] += 1

        text = string_data.decode("latin1", errors="replace")

        # Optional validation (disabled by default for performance)
        if validate and length > 20:
            printable_count = sum(1 for c in text if c.isprintable() or c in "\t\n\r")
            if printable_count < len(text) * 0.7:
                raise ValueError(
                    f"Data appears to be binary for {field_name} at position {start_pos}"
                )

        self._debug_log(field_name, text, start_pos)
        return text

    def read_raw_bytes(self, length: int, field_name: str = "") -> bytes:
        """Read raw bytes."""
        self._open_file()
        assert self.file is not None  # Type assertion for linter
        data = self.file.read(length)
        if len(data) != length:
            raise ValueError(
                f"Cannot read {length} bytes for {field_name} at position {self.position}: end of file"
            )

        self.position += length
        self.stats["bytes_read"] += length

        if self.debug_level >= 2 and field_name:
            print(f"    {field_name} (pos {self.position - length}): {length} bytes")

        return data

    def skip_raw_bytes(self, length: int, reason: str = ""):
        """Skip bytes efficiently."""
        self._open_file()
        assert self.file is not None  # Type assertion for linter
        self.file.seek(length, 1)  # Seek relative to current position
        self.position += length

        if self.debug_level >= 2 and reason:
            print(f"    Skipped {length} bytes: {reason} (now at pos {self.position})")

    def read_custom_field_definitions(self, version: int) -> List[CatalogCustomFieldProperties]:
        """Read custom field definitions with optimized parsing."""
        if self.debug_level >= 1:
            print("\n=== Reading Custom Fields Properties ===")

        # Read the properties that come before the field definitions
        column_settings = self.read_length_prefixed_string("column_settings")
        gui_properties = self.read_length_prefixed_string("gui_properties")

        if self.debug_level >= 1:
            print(f"  Column settings: {len(column_settings)} chars")
            print(f"  GUI properties: {len(gui_properties)} chars")

        # Read field count and definitions
        count = self.read_int32_le("custom_fields_count")
        custom_fields = []

        for i in range(count):
            if self.debug_level >= 1:
                print(f"  Reading custom field {i + 1}/{count}")

            # Read field definition efficiently
            field_tag = self.read_length_prefixed_string(f"field_tag_{i}")
            field_name = self.read_length_prefixed_string(f"field_name_{i}")

            # Version-specific fields
            if version >= 41:
                self.read_length_prefixed_string(
                    f"field_ext_{i}"
                )  # field_ext - read but don't store

            field_type = self.read_length_prefixed_string(f"field_type_{i}")
            default_value = self.read_length_prefixed_string(f"default_value_{i}")

            if version >= 41:
                self.read_length_prefixed_string(
                    f"media_info_{i}"
                )  # media_info - read but don't store

            self.read_bool_byte(f"multi_values_{i}")  # multi_values - read but don't store

            if version >= 41:
                self.read_raw_bytes(
                    4, f"multi_values_sep_{i}"
                )  # multi_values_sep - read but don't store
                self.read_bool_byte(
                    f"multi_values_rmp_{i}"
                )  # multi_values_rmp - read but don't store
                self.read_bool_byte(
                    f"multi_values_patch_{i}"
                )  # multi_values_patch - read but don't store

            self.read_bool_byte(
                f"excluded_in_scripts_{i}"
            )  # excluded_in_scripts - read but don't store
            self.read_length_prefixed_string(
                f"gui_properties_field_{i}"
            )  # gui_properties_field - read but don't store

            # Handle list values efficiently
            if field_type == "ftList":
                if self.debug_level >= 2:
                    print(f"    Field {i} is a list type, reading list values...")

                list_values_count = self.read_int32_le(f"list_values_count_{i}")

                # Read list values without storing them (optimization)
                for j in range(list_values_count):
                    self.read_length_prefixed_string(
                        f"list_value_{i}_{j}"
                    )  # list_value - read but don't store

                if version >= 41:
                    self.read_bool_byte(
                        f"list_auto_add_{i}"
                    )  # list_auto_add - read but don't store
                    self.read_bool_byte(f"list_sort_{i}")  # list_sort - read but don't store
                    self.read_bool_byte(
                        f"list_auto_complete_{i}"
                    )  # list_auto_complete - read but don't store
                    self.read_bool_byte(
                        f"list_use_catalog_values_{i}"
                    )  # list_use_catalog_values - read but don't store

            custom_fields.append(
                CatalogCustomFieldProperties(field_tag, field_name, field_type, default_value)
            )

        return custom_fields

    def read_movie_custom_field_values(
        self, custom_fields_properties: List[CatalogCustomFieldProperties]
    ) -> Dict[str, str]:
        """Read custom field values efficiently."""
        custom_fields = {}

        for i, field_prop in enumerate(custom_fields_properties):
            field_value = self.read_length_prefixed_string(f"custom_field_{i}_value")
            custom_fields[field_prop.field_tag] = field_value

            if self.debug_level >= 2:
                print(f"    Custom field '{field_prop.field_tag}': '{field_value}'")

        return custom_fields

    def read_embedded_movie_picture(self) -> Optional[CatalogMoviePicture]:
        """Read picture data efficiently."""
        pic_path = self.read_length_prefixed_string("pic_path")
        pic_size = self.read_int32_le("pic_size")

        if pic_size > 0:
            if self.debug_level >= 2:
                print(f"    Reading {pic_size} bytes of picture data from path '{pic_path}'")

            pic_data = self.read_raw_bytes(pic_size, "pic_data")
            extension = ""
            if pic_path and "." in pic_path:
                extension = "." + pic_path.split(".")[-1].lower()

            return CatalogMoviePicture(pic_path, pic_data, extension)
        else:
            if self.debug_level >= 2:
                print(f"    No picture data (size: {pic_size}, path: '{pic_path}')")
            return None

    def read_movie_properties(self, version: int) -> Dict[str, str]:
        """Read movie properties efficiently."""
        if self.debug_level >= 1:
            print("\n=== Reading Movie Properties ===")

        properties = {
            "owner_name": self.read_length_prefixed_string("owner_name"),
            "owner_mail": self.read_length_prefixed_string("owner_mail"),
        }

        # ICQ field was removed in version 35
        if version < 35:
            self.read_length_prefixed_string("icq_field_deprecated")

        properties.update(
            {
                "owner_site": self.read_length_prefixed_string("owner_site"),
                "description": self.read_length_prefixed_string("description"),
            }
        )

        return properties

    def read_movie_sequential(
        self,
        version: int,
        custom_fields_properties: Optional[List[CatalogCustomFieldProperties]] = None,
    ) -> Optional[CatalogMovie]:
        """Read a single movie record with parsing."""
        start_pos = self.position

        try:
            # Read integer fields efficiently
            movie_number = self.read_int32_le("number")
            date_added = self.read_int32_le("date_added")

            # Version-specific fields
            date_watched = 0
            user_rating = 0
            if version >= 42:
                date_watched = self.read_int32_le("date_watched")
                user_rating = self.read_int32_le("user_rating")

            rating = self.read_int32_le("rating")
            if version < 35 and rating != -1:
                rating = rating * 10

            year = self.read_int32_le("year")
            length = self.read_int32_le("length")
            video_bitrate = self.read_int32_le("video_bitrate")
            audio_bitrate = self.read_int32_le("audio_bitrate")
            disks = self.read_int32_le("disks")

            color_tag = 0
            if version >= 41:
                color_tag = self.read_int32_le("color_tag")
                color_tag = color_tag % 13

            checked = self.read_bool_byte("checked")

            # Read string fields efficiently (no validation for performance)
            media = self.read_length_prefixed_string("media")

            media_type = ""
            source = ""
            if version >= 33:
                media_type = self.read_length_prefixed_string("media_type")
                source = self.read_length_prefixed_string("source")

            borrower = self.read_length_prefixed_string("borrower")
            original_title = self.read_length_prefixed_string("original_title")
            translated_title = self.read_length_prefixed_string("translated_title")
            director = self.read_length_prefixed_string("director")
            producer = self.read_length_prefixed_string("producer")

            writer = ""
            composer = ""
            if version >= 42:
                writer = self.read_length_prefixed_string("writer")
                composer = self.read_length_prefixed_string("composer")

            country = self.read_length_prefixed_string("country")
            category = self.read_length_prefixed_string("category")

            certification = ""
            if version >= 42:
                certification = self.read_length_prefixed_string("certification")

            actors = self.read_length_prefixed_string("actors")
            url = self.read_length_prefixed_string("url")
            description = self.read_length_prefixed_string("description")
            comments = self.read_length_prefixed_string("comments")

            file_path = ""
            if version >= 42:
                file_path = self.read_length_prefixed_string("file_path")

            video_format = self.read_length_prefixed_string("video_format")
            audio_format = self.read_length_prefixed_string("audio_format")
            resolution = self.read_length_prefixed_string("resolution")
            framerate = self.read_length_prefixed_string("framerate")
            languages = self.read_length_prefixed_string("languages")
            subtitles = self.read_length_prefixed_string("subtitles")
            size = self.read_length_prefixed_string("size")

            # Read picture and custom fields
            picture = self.read_embedded_movie_picture()

            custom_fields = {}
            if version >= 40 and custom_fields_properties:
                custom_fields = self.read_movie_custom_field_values(custom_fields_properties)

            extras = []
            if version >= 42:
                extras = self.read_movie_extras()

            self.stats["movies_read"] += 1

            if self.debug_level >= 1:
                print(
                    f"Successfully read movie {movie_number}: '{original_title or translated_title}'"
                )

            return CatalogMovie(
                number=movie_number,
                original_title=original_title,
                translated_title=translated_title,
                director=director,
                producer=producer,
                writer=writer,
                composer=composer,
                actors=actors,
                country=country,
                year=year,
                length=length,
                category=category,
                certification=certification,
                url=url,
                description=description,
                comments=comments,
                user_rating=user_rating,
                rating=rating,
                date_watched=date_watched,
                date_added=date_added,
                checked=checked,
                color_tag=color_tag,
                media=media,
                media_type=media_type,
                source=source,
                borrower=borrower,
                file_path=file_path,
                video_format=video_format,
                video_bitrate=video_bitrate,
                audio_format=audio_format,
                audio_bitrate=audio_bitrate,
                resolution=resolution,
                framerate=framerate,
                languages=languages,
                subtitles=subtitles,
                size=size,
                disks=disks,
                picture=picture,
                custom_fields=custom_fields,
                extras=extras,
            )

        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error reading movie at position {start_pos}: {e}")
            return None

    def read_movie_extras(self) -> List[CatalogMovieExtra]:
        """Read movie extras efficiently."""
        extras = []
        extras_start_pos = self.position

        try:
            extras_count = self.read_int32_le("extras_count")
            if self.debug_level >= 2:
                print(f"    Reading {extras_count} extras")

            if extras_count < 0 or extras_count > 1000:
                if self.debug_level >= 1:
                    print(f"    Suspicious extras count: {extras_count}, skipping extras")
                return extras

            for i in range(extras_count):
                if self.debug_level >= 2:
                    print(f"    Reading extra {i + 1}/{extras_count} at position {self.position}")

                checked = self.read_bool_byte(f"extra_{i}_checked")
                tag = self.read_length_prefixed_string(f"extra_{i}_tag")
                title = self.read_length_prefixed_string(f"extra_{i}_title")
                category = self.read_length_prefixed_string(f"extra_{i}_category")
                url = self.read_length_prefixed_string(f"extra_{i}_url")
                description = self.read_length_prefixed_string(f"extra_{i}_description")
                comments = self.read_length_prefixed_string(f"extra_{i}_comments")
                created_by = self.read_length_prefixed_string(f"extra_{i}_created_by")

                picture = self.read_embedded_movie_picture()

                extra = CatalogMovieExtra(
                    checked=checked,
                    tag=tag,
                    title=title,
                    category=category,
                    url=url,
                    description=description,
                    comments=comments,
                    created_by=created_by,
                    picture=picture,
                )
                extras.append(extra)

                if self.debug_level >= 2:
                    pic_info = (
                        f" with {picture.size} byte {picture.extension} image"
                        if picture
                        else " (no image)"
                    )
                    print(f"    Extra {i + 1}: '{title}' ({category}){pic_info}")

        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error reading extras at position {extras_start_pos}: {e}")

        return extras

    def read_all_movies_sequential(
        self,
        version: int,
        custom_fields_properties: Optional[List[CatalogCustomFieldProperties]] = None,
    ) -> List[CatalogMovie]:
        """Read all movies with optimized sequential parsing."""
        movies = []

        if self.debug_level >= 1:
            print(f"\n=== Reading Movies Sequentially (Version {version}) ===")

        movie_count = 0
        while self.position < self.file_size:
            movie = self.read_movie_sequential(version, custom_fields_properties)
            if movie is None:
                break

            movies.append(movie)
            movie_count += 1

            # Progress reporting (less frequent for performance)
            if self.debug_level >= 1 and movie_count % 500 == 0:
                print(f"Read {movie_count} movies... ({self.position / self.file_size * 100:.1f}%)")

        if self.debug_level >= 1:
            print(f"Successfully read {len(movies)} movies sequentially")
            print(
                f"Statistics: {self.stats['strings_read']:,} strings, {self.stats['bytes_read']:,} bytes"
            )

        return movies

    def read_full_catalog_sequential(self) -> AntMovieCatalog:
        """Read the complete catalog with optimized parsing."""
        if self.debug_level >= 1:
            print(f"Reading AMC catalog with optimized parser (version {self.version})")

        # Read movie properties
        movie_properties = self.read_movie_properties(self.version)
        if self.debug_level >= 1:
            print(f"Owner: {movie_properties.get('owner_name', 'Unknown')}")

        # Read custom fields properties
        custom_fields_properties = []
        if self.version >= 40:
            custom_fields_properties = self.read_custom_field_definitions(self.version)

        # Movies start immediately after custom fields
        if self.debug_level >= 1:
            print(f"Movies start at position: {self.position}")

        # Read all movies
        movies = self.read_all_movies_sequential(self.version, custom_fields_properties)

        return AntMovieCatalog(
            version=self.version,
            header=self.header,
            custom_fields_properties=custom_fields_properties,
            movies=movies,
        )

    def close(self):
        """Close the file handle."""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Database Export Functions
# ============================================================================


def export_catalog_to_database(
    catalog: AntMovieCatalog, sqlite_path: str, movie_properties: Optional[dict] = None
):
    """
    Export complete AMC catalog to SQLite database including all data:
    - Catalog properties/owner info
    - Movies with all fields and images
    - Custom fields and their values
    - Movie extras without images (they are not saved in the database)
    """
    print(f"Exporting complete catalog to SQLite: {sqlite_path}")

    engine = create_engine(f"sqlite:///{sqlite_path}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # 1. Export catalog properties
        properties_orm = CatalogPropertiesORM(
            version=catalog.version,
            header=catalog.header,
            owner_name=movie_properties.get("owner_name", "") if movie_properties else "",
            owner_mail=movie_properties.get("owner_mail", "") if movie_properties else "",
            owner_site=movie_properties.get("owner_site", "") if movie_properties else "",
            description=movie_properties.get("description", "") if movie_properties else "",
        )
        session.add(properties_orm)

        # 2. Export custom field definitions
        custom_field_count = 0
        for custom_field in catalog.custom_fields_properties:
            session.add(
                CatalogCustomFieldORM(
                    tag=custom_field.field_tag,
                    name=custom_field.field_name,
                    type=custom_field.field_type,
                    default_value=custom_field.default_value,
                )
            )
            custom_field_count += 1

        # 3. Export movies with all data
        movie_count = 0
        custom_field_value_count = 0
        extra_count = 0
        movie_image_count = 0
        extra_image_count = 0

        for movie in catalog.movies:
            # Main movie data
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
                user_rating=movie.user_rating,
                rating=movie.rating,
                date_watched=movie.date_watched,
                date_added=movie.date_added,
                checked=movie.checked,
                color_tag=movie.color_tag,
                media=movie.media,
                media_type=movie.media_type,
                source=movie.source,
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
                # Movie image data
                picture_path=movie.picture.pic_path if movie.picture else None,
                picture_data=movie.picture.pic_data
                if movie.picture and movie.picture.pic_data
                else None,
                picture_extension=movie.picture.extension if movie.picture else None,
            )
            session.add(movie_orm)
            movie_count += 1

            if movie.picture and movie.picture.pic_data:
                movie_image_count += 1

            # Custom field values
            for field_tag, field_value in movie.custom_fields.items():
                session.add(
                    CatalogMovieCustomFieldORM(
                        movie_number=movie.number,
                        field_tag=field_tag,
                        field_value=field_value,
                    )
                )
                custom_field_value_count += 1

            # Movie extras
            for extra in movie.extras:
                extra_orm = CatalogMovieExtraORM(
                    movie_number=movie.number,
                    checked=extra.checked,
                    tag=extra.tag,
                    title=extra.title,
                    category=extra.category,
                    url=extra.url,
                    description=extra.description,
                    comments=extra.comments,
                    created_by=extra.created_by,
                    # Extra image data
                    picture_path=extra.picture.pic_path if extra.picture else None,
                    picture_data=extra.picture.pic_data
                    if extra.picture and extra.picture.pic_data
                    else None,
                    picture_extension=extra.picture.extension if extra.picture else None,
                )
                session.add(extra_orm)
                extra_count += 1

                if extra.picture and extra.picture.pic_data:
                    extra_image_count += 1

        # Commit all data
        session.commit()

    # Print summary
    print("Database export complete!")
    print("  - Catalog properties: 1")
    print(f"  - Custom field definitions: {custom_field_count}")
    print(f"  - Movies: {movie_count}")
    print(f"  - Custom field values: {custom_field_value_count}")
    print(f"  - Movie extras: {extra_count}")
    print(f"  - Movie images: {movie_image_count}")
    print(f"  - Extra images: {extra_image_count}")
    print(f"  - Total images in database: {movie_image_count + extra_image_count}")


def extract_all_embedded_images(catalog: AntMovieCatalog, output_directory: Path):
    """
    Extract ALL embedded images from catalog:
    - Movie poster/cover images
    - Custom field images (if any)
    - Movie extra images
    """
    output_directory.mkdir(exist_ok=True)

    movie_image_count = 0
    extra_image_count = 0

    print(f"Extracting all embedded images to: {output_directory}")

    for movie in catalog.movies:
        # Extract movie poster/cover image
        if movie.picture and movie.picture.pic_data:
            title = movie.original_title or movie.translated_title or f"movie_{movie.number}"
            sanitized_title = sanitize_filename(title)

            # Determine file extension
            extension = movie.picture.extension or ".jpg"
            if not extension.startswith("."):
                extension = f".{extension}"

            image_filename = f"{movie.number:04d}_{sanitized_title}_poster{extension}"
            image_filepath = output_directory / image_filename

            with open(image_filepath, "wb") as f:
                f.write(movie.picture.pic_data)

            print(f"  Movie {movie.number}: {image_filename} ({movie.picture.size:,} bytes)")
            movie_image_count += 1

        # Extract movie extra images
        for i, extra in enumerate(movie.extras):
            if extra.picture and extra.picture.pic_data:
                title = movie.original_title or movie.translated_title or f"movie_{movie.number}"
                sanitized_title = sanitize_filename(title)
                extra_title = sanitize_filename(extra.title) if extra.title else f"extra_{i + 1}"

                # Determine file extension
                extension = extra.picture.extension or ".jpg"
                if not extension.startswith("."):
                    extension = f".{extension}"

                image_filename = f"{movie.number:04d}_{sanitized_title}_{extra_title}{extension}"
                image_filepath = output_directory / image_filename

                with open(image_filepath, "wb") as f:
                    f.write(extra.picture.pic_data)

                print(
                    f"  Extra {movie.number}-{i + 1}: {image_filename} ({extra.picture.size:,} bytes)"
                )
                extra_image_count += 1

    total_images = movie_image_count + extra_image_count
    print("\nImage extraction complete!")
    print(f"  - Movie images: {movie_image_count}")
    print(f"  - Extra images: {extra_image_count}")
    print(f"  - Total images extracted: {total_images}")

    return total_images


# ============================================================================
# Utility Functions
# ============================================================================


def delphi_date_to_datetime(delphi_days):
    """Convert Delphi TDateTime (days since 1899-12-30) to Python datetime."""
    try:
        base_date = datetime(1899, 12, 30)
        return base_date + timedelta(days=delphi_days)
    except (ValueError, OverflowError):
        return None


def sanitize_filename(title: str) -> str:
    """Sanitize a movie title for use as a filename."""
    return (
        "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip().replace(" ", "_")
    )


def parse_amc_file_optimized(file_path: str, debug_level: int = 0) -> AntMovieCatalog:
    """
    Parse an AMC file.

    Args:
        file_path: Path to the AMC file
        debug_level: 0=no debug, 1=basic info, 2=detailed debug

    Returns:
        AntMovieCatalog object with parsed data
    """
    with AntMovieCatalogReader(file_path, debug_level=debug_level) as reader:
        return reader.read_full_catalog_sequential()


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="AMC Parse")
    parser.add_argument("amc_file", help="Path to AMC file")
    parser.add_argument("--sqlite-db", type=str, help="Export to SQLite database file")
    parser.add_argument(
        "--extract-images",
        type=str,
        metavar="DIR",
        help="Extract embedded images (movie + extras) to directory",
    )
    parser.add_argument(
        "--debug",
        "-d",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=basic, 2=detailed",
    )
    parser.add_argument("--stats", "-s", action="store_true", help="Show performance statistics")

    args = parser.parse_args()

    if not Path(args.amc_file).exists():
        print(f"Error: AMC file '{args.amc_file}' not found")
        sys.exit(1)

    try:
        print(f"Parsing AMC file: {args.amc_file}")
        start_time = datetime.now()

        # Parse the AMC file with optimized parser
        with AntMovieCatalogReader(args.amc_file, debug_level=args.debug) as reader:
            # Read movie properties first (needed for database)
            movie_properties = reader.read_movie_properties(reader.version)

            # Reset to read full catalog
            assert reader.file is not None  # Type assertion for linter
            reader.file.seek(65)  # Reset to start of data section
            reader.position = 65
            catalog = reader.read_full_catalog_sequential()

        end_time = datetime.now()

        print(f"Successfully parsed {len(catalog.movies)} movies")
        print(f"  - Custom fields: {len(catalog.custom_fields_properties)}")
        print(f"  - Version: {catalog.version}")
        print(f"  - Parse Time: {(end_time - start_time).total_seconds():.2f} seconds")

        if args.stats:
            print("\n=== Performance Statistics ===")
            # Statistics would be available if we kept the reader instance

        # Export to SQLite database
        if args.sqlite_db:
            export_catalog_to_database(catalog, args.sqlite_db, movie_properties)

        # Extract all images
        if args.extract_images:
            extract_all_embedded_images(catalog, Path(args.extract_images))

        if not args.sqlite_db and not args.extract_images:
            print("\nNo output specified. Use --sqlite-db or --extract-images")
            print(
                "Example: python amc_complete_standalone.py movie.amc --sqlite-db movies.db --extract-images ./images/"
            )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
