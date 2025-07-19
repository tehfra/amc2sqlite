A python script for parsing Ant Movie Catalog databases, that can also export to an SQLite database.

Uses `sqlalchemy`:
```bash
pip install sqlalchemy
```

### Basic Usage
This is basically a dry-run, it does not export anything:

```bash
python amc_parse.py your_database.amc
```

### Extract Embedded Images

```bash
python amc_parse.py your_database.amc --extract-images ./images/
```

### Export to SQLite

```bash
python amc_parse.py your_database.amc --sqlite-db movies.sqlite
```

### Debug Mode

```bash
python amc_parse.py your_database.amc --debug 2
```

### Combined Options

```bash
python amc_parse.py your_database.amc \
    --extract-images ./images/ \
    --sqlite-db movies.sqlite \
    --debug 1
```
