SET threads=32;
drop table if exists patient_metadata;
drop table if exists dataset;

create table patient_metadata as (
    SELECT v_pnr_enc as pid,
		c_kon as sex,
		d_foddato as birthdate, -- may be redundant because this could be already type date
		end_of_data as outcome_date,  -- will be overwritten in add_metadata step for indivduals with other censorings
    (CASE WHEN random_split_index  < 0.6 THEN 'train'
        WHEN random_split_index  between 0.6 and 0.8 THEN 'dev'
        WHEN random_split_index  > 0.8 THEN 'test' END) as split_group
    FROM (
		SELECT *
		, random() as random_split_index
		FROM t_person
		WHERE c_status in ('01','90', '80') 
			and d_foddato is not NULL
	) as foo
);

create table dataset as (

WITH  diag as (
    SELECT PERSON_ID as pid
	    , 'diag' as modality
        , d_inddto as admit_date
		, c_diag as code
    FROM (
		SELECT *
        FROM t_diag_adm
        WHERE c_diagtype in ('A','B','C')
        ) as foo
), 
prescription as (
    SELECT PERSON_ID as pid, 
    'prescription' as modality,
	eksd as admit_date,
	ATC as code
	FROM lmdb
	where atc!=''
), 
service_registry as (
	select pid
		, 'ydelse' as modality
		, date as admit_date
		, fullcode_grouped as code
	from ydelse
	inner join (
		select fullcode_raw, fullcode_grouped  
		from ydelse_mapping 
		) as ydelse_mapping
	on (ydelse.code=ydelse_mapping.fullcode_raw)
), 
all_events as (
	select * from diag union all 
	select * from prescription union all 
	select * from service_registry
) 
select pid, modality, code, date_trunc('day',admit_date) as admit_date from all_events

);
