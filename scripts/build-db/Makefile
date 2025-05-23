## --------------------------------------------------------------------------
## Description: Creation of the dataset
## -------------------------------------------------------------------------- 

# Set the following parameters to defaults
# These will be overwritten by settings passed to the makefile
OUTCOME_TYPE := T2D


SCRIPT_DIR := scripts/build-db
DATA_DIR := data
SCHEMA := diabetes
DBNAME := $(DATA_DIR)/$(SCHEMA)/duckdb.db


all: clean_db create_tables build_dataset metadata create_vocabulary

clean_db:
	rm -f $(DBNAME)

create_tables:
	mkdir -p $(DATA_DIR)/$(SCHEMA)
	duckdb $(DBNAME) -c "CREATE TABLE dummy_table (id INTEGER);"
	duckdb $(DBNAME) -c "DROP TABLE dummy_table;"
	duckdb $(DBNAME) < $(SCRIPT_DIR)/create_tables.sql

build_dataset:
	duckdb $(DBNAME) < $(SCRIPT_DIR)/build_dataset.sql


metadata: fetch_dates create_registry add_metadata

fetch_dates:
	duckdb $(DBNAME) < $(SCRIPT_DIR)/diabetes_dates.sql

create_registry:
	python $(SCRIPT_DIR)/diabetes_registry.py --database $(DBNAME) --outcome-type $(OUTCOME_TYPE)

add_metadata:
	python $(SCRIPT_DIR)/add_metadata.py --database $(DBNAME)

create_vocabulary:
	python $(SCRIPT_DIR)/create_vocabulary.py --duckdb-database $(DBNAME)

