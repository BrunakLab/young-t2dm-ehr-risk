import json
from typing import Optional

import duckdb
import pandas as pd

UNK_CODE = "<UNK>"
PAD_CODE = "<PAD>"
NONE_CODE = "<NONE>"


class ModalityType:
    DIAG = "diag"
    PRESCRIPTION = "prescription"
    YDELSE = "ydelse"


ModalityIndex = {
    ModalityType.DIAG: 0,
    ModalityType.PRESCRIPTION: 1,
    ModalityType.YDELSE: 2,
}


class CodeType:
    ICD10 = "icd10"
    ICD8 = "icd8"
    ATC = "atc"
    YDELSE = "ydelse"
    NONE = "none"


class CodeParser:
    def __init__(
        self,
        atc_level: Optional[int] = None,
        diag_level: Optional[int] = None,
        ydelse_level: Optional[int] = None,
    ) -> None:
        """
        Parses codes according to the truncation_levels specified.
        """
        self.atc_level = atc_level
        self.diag_level = diag_level
        self.ydelse_level = ydelse_level

    def parse_code(
        self, event: str, modality: ModalityType, truncation_level: Optional[int] = None
    ):
        """
        Parses the code. Truncation Level can be specified to overwrite behaviour from the initializer
        Example truncations:
            DE102: level 1 = DE, level 2 = DE1, level 3 = DE10
            A10BA: level 0 = A, level 1 = A1 (This does not make sense), level 2 = A10, level 3 = A10B, level 4 = A10BA
            820642: level 1 = 82 (chapter), level 5 = 820642 (full code)
        """

        if modality == ModalityType.YDELSE:
            ydelse_level = self._validate_level(
                base_level=self.ydelse_level, level=truncation_level
            )
            return event[: ydelse_level + 1], CodeType.YDELSE

        elif modality == ModalityType.PRESCRIPTION:
            atc_level = self._validate_level(
                base_level=self.atc_level, level=truncation_level
            )
            return event[: atc_level + 1], CodeType.ATC  # A10BA --> level 4

        elif modality == ModalityType.DIAG:
            event = event.replace(".", "")
            diag_level = self._validate_level(
                base_level=self.diag_level, level=truncation_level
            )

            if (
                event[0] == "D" and not event[1].isdigit()
            ):  # this means it is a SKS event
                return event[: diag_level + 1], CodeType.ICD10
            elif event.isdigit():
                return event[:diag_level], CodeType.ICD8
            elif (
                event[0] == "Y" or event[0] == "E"
            ):  # TODO data - separate SKS or RPDR event by the data class not by filtering
                return event[: diag_level + 1], CodeType.ICD8
            else:
                return event[:diag_level], CodeType.ICD10
        else:
            print("Unknown modality : {}. Returning event unmodified".format(modality))
            return event, CodeType.NONE

    def _validate_level(self, base_level: Optional[int], level: Optional[int]) -> int:
        if level:
            return level
        elif base_level:
            return base_level
        raise ValueError(
            "Base level and level was not specifiec in initializer or function."
        )


class Vocabulary:
    """
    Container for vocabulary. Core datastructure is a dict for each modality with a specified truncation level. (Should be initialized currently with the from duckdb method)
    """

    def __init__(self, diag: dict, prescription: dict, ydelse: dict) -> None:
        self.diag_to_index = diag
        self.index_to_diag = {index: code for code, index in diag.items()}
        self.prescription_to_index = prescription
        self.index_to_prescription = {
            index: code for code, index in prescription.items()
        }
        self.ydelse_to_index = ydelse
        self.index_to_ydelse = {index: code for code, index in ydelse.items()}

    def code_to_index(self, code: str, modality: str):
        if modality == ModalityType.YDELSE:
            return self.ydelse_to_index.get(code, self.ydelse_to_index[UNK_CODE])
        if modality == ModalityType.PRESCRIPTION:
            return self.prescription_to_index.get(
                code, self.prescription_to_index[UNK_CODE]
            )
        if modality == ModalityType.DIAG:
            return self.diag_to_index.get(code, self.diag_to_index[UNK_CODE])

    def index_to_code(self, index: int, modality: str):
        if modality == ModalityType.YDELSE:
            return self.index_to_ydelse[index]
        if modality == ModalityType.PRESCRIPTION:
            return self.index_to_prescription[index]
        if modality == ModalityType.DIAG:
            return self.index_to_diag[index]

    def get_sizes(self):
        return {
            ModalityType.YDELSE: max(self.ydelse_to_index.values()) + 1,
            ModalityType.PRESCRIPTION: max(self.prescription_to_index.values()) + 1,
            ModalityType.DIAG: max(self.diag_to_index.values()) + 1,
        }

    def code_to_all_indexes(self, code: str):
        """
        Calls code_to_index for all modalities and returns a mapping. Useful for fetching general indexes such as padding
        """
        return {
            ModalityType.YDELSE: self.code_to_index(code, ModalityType.YDELSE),
            ModalityType.PRESCRIPTION: self.code_to_index(
                code, ModalityType.PRESCRIPTION
            ),
            ModalityType.DIAG: self.code_to_index(code, ModalityType.DIAG),
        }

    @classmethod
    def load_from_duckdb(cls, duckdb_database):
        conn = duckdb.connect(duckdb_database, read_only=True)
        vocab = conn.execute("SELECT * from vocab").df()
        split_vocabs = {
            ModalityType.DIAG: {},
            ModalityType.PRESCRIPTION: {},
            ModalityType.YDELSE: {},
        }

        for _, row in vocab.iterrows():
            split_vocabs[row["modality"]][row["code"]] = row["code_index"]
        return cls(**split_vocabs)

    @classmethod
    def create_from_duckdb(
        cls,
        duckdb_database: str,
        diag_level: int,
        atc_level: int,
        ydelse_level: int,
        table: Optional[str] = None,
    ):
        """
        Create a new vocabulary from duckdb using the specified parameters
        """

        conn = duckdb.connect(duckdb_database)

        if "code_index" in conn.execute("select * from dataset limit 1").df().columns:
            print(
                "Warning: Creating a new vocabulary that may not match the current made one. If you want to use the current defined vocab please use load_from_duckdb method. If updating vocab, then ignore this"
            )
        else:
            print(
                "Warning: If training a model now. The code_index column may not exist. Please check that this column exist."
            )
        print("Building new Vocabulary...")

        split_vocabs = {
            ModalityType.DIAG: {PAD_CODE: 0, UNK_CODE: 1, NONE_CODE: 2},
            ModalityType.PRESCRIPTION: {PAD_CODE: 0, UNK_CODE: 1, NONE_CODE: 2},
            ModalityType.YDELSE: {PAD_CODE: 0, UNK_CODE: 1, NONE_CODE: 2},
        }
        token_to_index = {
            ModalityType.DIAG: {PAD_CODE: 0, UNK_CODE: 1, NONE_CODE: 2},
            ModalityType.PRESCRIPTION: {PAD_CODE: 0, UNK_CODE: 1, NONE_CODE: 2},
            ModalityType.YDELSE: {PAD_CODE: 0, UNK_CODE: 1, NONE_CODE: 2},
        }

        extract_vocab_query = """select distinct code, modality 
                                      from dataset
                                      where pid in (select pid from patient_metadata where split_group = 'train')
                                      order by code
                                  """
        vocab = conn.execute(extract_vocab_query).df()
        code_parser = CodeParser(atc_level, diag_level, ydelse_level)
        vocab["token"], vocab["code_type"] = zip(
            *vocab.apply(
                lambda x: code_parser.parse_code(x["code"], x["modality"]), axis=1
            )
        )
        special_token_vocab = [
            {
                "code": code,
                "modality": modality,
                "token": code,
                "code_index": split_vocabs[modality][code],
            }
            for code in [PAD_CODE, UNK_CODE, NONE_CODE]
            for modality in [
                ModalityType.DIAG,
                ModalityType.PRESCRIPTION,
                ModalityType.YDELSE,
            ]
        ]

        # Filter icd8 codes for vocabulary
        vocab = vocab[vocab["code_type"] != CodeType.ICD8]

        for _, row in vocab.iterrows():
            # Ensure that two raw codes with same token map to same value.
            # Thereby we only need to map from raw code to index.
            if row["token"] in token_to_index[row["modality"]]:
                vocab_index = token_to_index[row["modality"]][row["token"]]
            else:
                vocab_index = len(token_to_index[row["modality"]])
                token_to_index[row["modality"]][row["token"]] = vocab_index

            split_vocabs[row["modality"]][row["code"]] = vocab_index

        vocab["code_index"] = vocab.apply(
            lambda row: split_vocabs[row["modality"]][row["code"]], axis=1
        )
        complete_vocab = pd.concat(
            [pd.DataFrame.from_records(special_token_vocab), vocab]
        )

        modality_index_df = pd.DataFrame.from_dict(
            ModalityIndex, orient="index", columns=["modality_index"]
        ).reset_index(names="modality")

        if table:
            query = f"""CREATE OR REPLACE TABLE {table} AS (
                    SELECT pid, admit_date, code, modality, token ,cast(modality_index as INT1) as modality_index, cast(code_index as INT2) as code_index
                    FROM (SELECT pid, admit_date, code, modality FROM {table})
                    LEFT JOIN (SELECT code, modality, token, code_index from complete_vocab) using (code, modality)
                    LEFT JOIN (SELECT distinct modality, modality_index from modality_index_df) using (modality)
                )
        """
            conn.execute(query)

        conn.execute(
            """CREATE OR REPLACE TABLE vocab AS (SELECT * from complete_vocab)"""
        )
        save_database_config(
            duckdb_database,
            {
                "diag_level": diag_level,
                "atc_level": atc_level,
                "ydelse_level": ydelse_level,
            },
        )

        return cls(**split_vocabs)

    def modality_to_index(self, modality: str):
        return ModalityIndex[modality]


def save_database_config(duckdb_database: str, config: dict[str, int]):
    json.dump(config, open(f"{duckdb_database}.config.json", "w"))
