SET threads=32;

CREATE OR REPLACE TABLE diabetes_dates as 
with first_de10 as (SELECT PERSON_ID as pid,
                               min(D_INDDTO) as de10_date,
                               count(*) as de10_count
                        FROM t_diag_adm 
                        WHERE SUBSTRING(C_diag,1,4) = 'DE10'
                              OR (SUBSTRING(C_diag,1,3) = '249')
                        GROUP BY PERSON_ID
      ),
      first_de11 as (SELECT PERSON_ID as pid,
                               min(D_INDDTO) as de11_date,
                               count(*) as de11_count
                        FROM t_diag_adm 
                        WHERE SUBSTRING(C_diag,1,4) = 'DE11'
                              OR (SUBSTRING(C_diag,1,3) = '250')
                        GROUP BY PERSON_ID
      ),
       first_other_diag as (SELECT PERSON_ID as pid,
                              min(D_INDDTO) as other_diag_date,
                               count(*) as other_diag_count
                        FROM t_diag_adm 
                        WHERE SUBSTRING(C_diag,1,4) IN ( 'DE12', 'DE13', 'DE14')
                        GROUP BY PERSON_ID
      ),

      first_pcos as (SELECT PERSON_ID as pid,
                          min(D_INDDTO) as pcos_date
                          FROM t_diag_adm
                          WHERE SUBSTRING(C_diag,1,5) IN ('DE282','61520','61521')
                          GROUP BY PERSON_ID
      ),
      first_gdm as (SELECT PERSON_ID as pid,
                         min(D_INDDTO) as gdm_date
                         FROM t_diag_adm
                         WHERE SUBSTRING(C_diag,1,4) = 'DO24' OR SUBSTRING(C_diag, 1,5) IN ('63474','Y6449')
                         GROUP BY PERSON_ID
      ),
      first_insulin as (SELECT PERSON_ID as pid,
                         min(eksd) as insulin_date,
                         count(*) as insulin_count
                         FROM lmdb
                         WHERE SUBSTRING(ATC,1,4) = 'A10A'
                         GROUP BY PERSON_ID),

      first_antidiabetic as (SELECT PERSON_ID as pid,
                         min(eksd) as antidiabetic_date,
                         count(*) as antidiabetic_count
                         FROM lmdb
                         WHERE SUBSTRING(ATC,1,4) = 'A10B'
                         GROUP BY PERSON_ID),
       first_footterapy as (SELECT pid,
                         min(date) as footterapy_date,
                         count(*) as footterapy_count,
                         FROM ydelse 
                         WHERE SUBSTRING(code,1,3) = '542'
                         GROUP BY pid),

       first_common_diags as (SELECT COALESCE(first_de10.pid, first_de11.pid) as pid, 
                             de10_date,
                             de10_count,
                             de11_date,
                             de11_count
                        FROM first_de10
                        FULL OUTER JOIN first_de11 ON first_de10.pid=first_de11.pid),
       
       first_diagnosis as (SELECT COALESCE(first_common_diags.pid, first_other_diag.pid) as pid, 
                             de10_date,
                             de10_count,
                             de11_date,
                             de11_count,
                             other_diag_date,
                             other_diag_count
                        FROM first_common_diags
                        FULL OUTER JOIN first_other_diag ON first_common_diags.pid=first_other_diag.pid),

      events_insulin as (SELECT COALESCE(first_diagnosis.pid, first_insulin.pid) as pid, 
                             de10_date,
                             de10_count,
                             de11_date,
                             de11_count,
                             other_diag_date,
                             other_diag_count,
                             insulin_date,
                             insulin_count
                        FROM first_diagnosis
                        FULL OUTER JOIN first_insulin ON first_diagnosis.pid=first_insulin.pid),
      
      first_event as (SELECT COALESCE(events_insulin.pid, first_antidiabetic.pid) as pid, 
                             de10_date,
                             de10_count,
                             de11_date,
                             de11_count,
                             other_diag_date,
                             other_diag_count,
                             insulin_date,
                             insulin_count,
                             antidiabetic_date,
                             antidiabetic_count
                        FROM events_insulin
                        FULL OUTER JOIN first_antidiabetic ON events_insulin.pid=first_antidiabetic.pid),

      events_plus_pcos as (SELECT COALESCE(first_event.pid, first_pcos.pid) as pid, 
                                  de10_date,
                                   de10_count,
                                   de11_date,
                                   de11_count,
                                   other_diag_date,
                                    other_diag_count,
                                  insulin_date,
                                  insulin_count,
                                  antidiabetic_date,
                                  antidiabetic_count,
                                  pcos_date
                           FROM first_event
                           FULL OUTER JOIN first_pcos ON first_event.pid=first_pcos.pid),

       events_plus_footterapy as (SELECT COALESCE(events_plus_pcos.pid, first_footterapy.pid) as pid, 
                                  de10_date,
                                   de10_count,
                                   de11_date,
                                   de11_count,
                                   other_diag_date,
                                    other_diag_count,
                                  insulin_date,
                                  insulin_count,
                                  antidiabetic_date,
                                  antidiabetic_count,
                                  pcos_date,
                                  footterapy_date,
                                  footterapy_count
                            FROM events_plus_pcos
                           FULL OUTER JOIN first_footterapy ON events_plus_pcos.pid=first_footterapy.pid),
      
      events_combined as (SELECT COALESCE(events_plus_footterapy.pid, first_gdm.pid) as pid, 
                                 de10_date,
                                 de10_count,
                                 de11_date,
                                 de11_count,
                                 other_diag_date,
                                 other_diag_count,
                                  insulin_date,
                                  insulin_count,
                                  antidiabetic_date,
                                  antidiabetic_count,
                                  pcos_date,
                                  footterapy_date,
                                  footterapy_count,
                                  gdm_date
                           FROM events_plus_footterapy
                           FULL OUTER JOIN first_gdm ON events_plus_footterapy.pid=first_gdm.pid)
      


SELECT events_combined.pid as pid,
       sex, 
       birthdate,
       de10_date,
       de10_count, 
       de11_date, 
       de11_count, 
       other_diag_date, 
       other_diag_count, 
       pcos_date,
       gdm_date, 
       footterapy_date, 
       footterapy_count,
       insulin_date, 
       insulin_count, 
       antidiabetic_date, 
       antidiabetic_count
FROM events_combined
INNER JOIN (SELECT v_pnr_enc as pid, C_KON as sex, D_FODDATO as birthdate FROM t_person GROUP BY v_pnr_enc, C_KON, D_FODDATO) as metadata ON events_combined.pid = metadata.pid;




