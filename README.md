A hacky Python script for converting an Ant Movie Catalog 4.2 database (not tested with any other version) to an SQLite database.
It is basically made to work with just IMDb imported data. If it works with custom and extras fields, I'm not complaining. If it works for you, good, but if not, hack away.

Uses `sqlalchemy` and `tqdm`.

#### Basic Usage
This is basically a dry-run, it does not export anything:

```bash
python amc2sqlite.py your_database.amc
```

#### Extract Embedded Images

```bash
python amc2sqlite.py your_database.amc --extract-images ./images/
```

#### Export to SQLite

```bash
python amc2sqlite.py your_database.amc --sqlite-path movies.sqlite
```

#### Debug Mode

```bash
python amc2sqlite.py your_database.amc --debug
```

#### Combined Options

```bash
python amc2sqlite.py your_database.amc \
    --extract-images ./images/ \
    --sqlite-path movies.sqlite \
    --debug
```

#### Pattern-Based Detection

The hacky part about this script. It uses pattern matching to locate movie records. Each movie record in an AMC database seems to follow a semi-consistent binary pattern:

```
.. .. 00 00 .. .. 00 00 .. .. 00 00 .. .. .. .. .. 00 00 00 .. .. .. .. .. .. 00 00 .. .. .. .. .. .. .. .. .. .. .. .. 00 00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 .. 00 00 00
```

Where:
- `..` = any byte (wildcard)
- `00` = must be zero byte
- `01` = must be 0x01 byte

After honing the pattern down to that particular one, it found all the 2021 movies in my database with no overmatching.

Once movie positions are identified, the script extracts from pattern offsets:

- Movie ID: Bytes 0-1 (little-endian)
- Date Added: Bytes 4-7 (Delphi TDateTime format)
- Rating: Byte 16 (IMDB rating × 10)
- Year: Bytes 20-21 (little-endian uint16)
- Length: Bytes 24-25 (little-endian uint16)
- Original Title Field Length: Byte 61

and then starting at offset +65 from pattern position:

- Original title (using the field length from before)
- Translated title, director, producer, etc. (length-prefixed strings)
- Embedded images (if present)
- Custom field values, I hope
