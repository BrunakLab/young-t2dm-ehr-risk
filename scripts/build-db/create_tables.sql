SET threads=32;
cd ./../../data

drop table if exists t_person;
create table t_person as 
    select c_status
        , end_of_data
        , v_pnr_enc
        , d_status_hen_start
        , d_foddato
        , c_kon
    from "t_person.tsv";

drop table if exists t_diag_adm;
create table t_diag_adm as 
    select person_id
        , d_inddto
        , c_diag
        , c_diagtype
        , c_sgh
        from "t_diag_adm.tsv"

drop table if exists lmdb;
create table lmdb as 
    select strptime(CAST(eksd as varchar),'%Y%m%d') as eksd 
        , atc
        , person_id
    from "prescription.tsv"

drop table if exists ydelse;
create table ydelse as 
    select pid
        , date
        , code 
    from "ydelse.tsv"
        and code is not NULL;

drop table if exists ydelse_mapping;
create table ydelse_mapping as 
    select * 
    from read_csv_auto('documentation/ydelse_mapping.tsv', header=True)
    where fullcode_grouped is not null;

