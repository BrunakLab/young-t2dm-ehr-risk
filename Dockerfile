FROM python:3.10.9-bullseye

################################################################################

RUN pip3 install \
    duckdb==0.10.0 \
    polars==0.19.12 \
    torch==1.13.1 \
    torchdata==0.5.1 \
    matplotlib==3.6.3 \
    seaborn==0.12.2 \
    captum==0.5.0 \
    fuzzywuzzy \
    Levenshtein


# add the duckdb extensions
RUN python3 <<EOF
import duckdb 
duckdb.install_extension('postgres')
duckdb.install_extension('fts')
duckdb.install_extension('parquet')
EOF

RUN wget https://github.com/duckdb/duckdb/releases/download/v0.10.0/duckdb_cli-linux-amd64.zip && \
    unzip duckdb_cli-linux-amd64.zip && \
    mv duckdb /usr/local/bin/ && \
    rm duckdb_cli-linux-amd64.zip

# add the duckdb extensions
RUN duckdb <<EOF
install postgres;
install fts;
install parquet;
EOF
