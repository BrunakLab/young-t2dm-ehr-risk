library(DBI)
library(duckdb)
library(tidyverse)
library(patchwork)
library(lubridate)


load_parquet <- function(path) {
    # Connect to an in-memory DuckDB database
    con <- dbConnect(duckdb::duckdb())
    
    # Construct the query to read the Parquet file
    query <- paste0("SELECT * FROM read_parquet('", path, "')")
    
    # Execute the query and store the result in a data frame
    df <- dbGetQuery(con, query)
    
    # Disconnect from the database
    dbDisconnect(con, shutdown = TRUE)
    
    # Return the data frame
    return(df)
}
color_mapping_modality <- list(
  diagnosis = "firebrick",    # Red
  prescription = "#FFA500", # Orange
  service = "cornflowerblue"       # Blue
)
  
positive_attributions <- load_parquet('data/diabetes/attributions/positive_attributions.parquet')
positive_attributions_patients <-  load_parquet('data/diabetes/attributions/positive_attributions_patients.parquet')
negative_attributions_patients <-  load_parquet('data/diabetes/attributions/negative_attributions_patients.parquet')
positive_attributions_patients_yearly <-  load_parquet('data/diabetes/attributions/positive_attributions_patients_yearly.parquet')
positive_attributions_yearly <-  load_parquet('data/diabetes/attributions/positive_attributions_yearly.parquet')


chapter_attributions <- positive_attributions %>% 
  mutate(abs_attribution=abs(patient_average_attribution),
                             modality=case_match(modality,
                                                 "diag" ~ "diagnosis",
                                                 "ydelse" ~ "service",
                                                 .default=modality)) %>% 
  group_by(chapter, modality) %>% 
  summarise(attribution = sum(abs_attribution)) %>% 
  ungroup()

code_attributions <- positive_attributions %>% 
  mutate(abs_attribution=abs(patient_average_attribution),
         modality=case_match(modality,
                             "diag" ~ "diagnosis",
                             "ydelse" ~ "service",
                             .default=modality)) %>% 
  group_by(chapter, modality, description, code) %>% 
  summarise(attribution = sum(abs_attribution), patient_counts = sum(patient_counts)) %>% 
  ungroup() %>% 
  filter(patient_counts >= 200)


modality_attributions <- chapter_attributions %>% 
  group_by(modality) %>% 
  summarise(attribution = sum(attribution)) %>% 
  mutate(relative_attribution=attribution/sum(attribution),
         x_label='All')


# Plot chapter and modality level attributions ----------------------------
ggplot(modality_attributions, aes(x=x_label,y=relative_attribution, fill=modality %>% str_to_title()))+
  geom_col(position="stack", color="black")+
  scale_fill_manual(values=color_mapping_modality %>% str_to_title())+
  geom_text(aes(label=modality %>% str_to_title()), position = position_stack(vjust=0.5))+
  theme_bw()+
  theme(legend.position = "none", axis.ticks.x= element_blank(), axis.text.x = element_blank())+
  labs(x="", y="Relative Attribution")

chapter_attributions <- chapter_attributions %>% 
  mutate(x_label="All", relative_attribution=attribution/sum(attribution)) %>%
  arrange(relative_attribution)

top_25_chapters <- chapter_attributions %>% slice_max(order_by=relative_attribution, n=25) %>% select(chapter, modality)

chapter_to_modality_color <- setNames(sapply(top_25_chapters$modality, function(x) color_mapping_modality[[x]]), top_25_chapters$chapter)

chapter_attributions%>%
  mutate(chapter_label = if_else(chapter %in% top_25_chapters$chapter, chapter, "Other"),
         ordering = if_else(chapter_label != "Other", rank(relative_attribution, ties.method = 'min'), 0)) %>%
  group_by(chapter_label, x_label,ordering) %>% 
  summarise(relative_attribution=sum(relative_attribution)) %>% 
ggplot(aes(x=x_label,y=relative_attribution, fill=fct_reorder(chapter_label, desc(ordering))))+
  geom_col(width=0.5,position="stack", color="black")+
  scale_fill_manual(values=chapter_to_modality_color)+
  geom_text(aes(x=1.3,label=chapter_label %>% str_to_title()), position = position_stack(vjust=0.5), hjust=0,)+
  theme_bw()+
  theme(legend.position = "none", axis.ticks.x= element_blank(), axis.text.x = element_blank())+
  coord_cartesian(xlim = c(1.2, 8))+
  labs(x="", y="Relative Absolute Attribution")



# Chapter Attributions Yearly ---------------------------------------------

chapter_attributions_yearly <- positive_attributions_yearly %>% 
  mutate(abs_attribution=abs(patient_average_attribution),
         modality=case_match(modality,
                             "diag" ~ "diagnosis",
                             "ydelse" ~ "service",
                             .default=modality)) %>% 
  group_by(chapter, modality, years_to_diagnosis) %>% 
  summarise(attribution = sum(abs_attribution)) %>% 
  ungroup()



chapter_attributions_yearly_plot <- chapter_attributions_yearly %>%
  group_by(chapter, years_to_diagnosis, modality) %>% 
  summarise(attribution=sum(attribution))

chapter_year_diff <- chapter_attributions_yearly_plot %>% 
  pivot_wider(values_from=attribution, names_from=years_to_diagnosis) %>% 
  mutate(diff=`0` - `4`)

diff_break <- sort(chapter_year_diff$diff,decreasing = TRUE)[15]

chapter_attributions_yearly_plot %>% 
  filter(chapter %in% top_25_chapters$chapter) %>% 
  inner_join(chapter_year_diff %>% select(chapter, diff), by="chapter") %>% 
  mutate(chapter_label=if_else(years_to_diagnosis == 4 & diff >= diff_break & chapter %in% top_25_chapters$chapter, chapter, ""),
         alpha = if_else(diff >= diff_break, 1, 0.5)) %>% 
  ggplot(aes(x=as_factor(years_to_diagnosis),y=attribution, group=chapter, color=modality, alpha=alpha))+
  geom_line(linewidth=2)+
  geom_point(size=3)+
  geom_text_repel(aes(label=chapter_label %>% str_to_title()), hjust=0, nudge_x = 0.1, direction = "y")+
  scale_color_manual(values=color_mapping_modality %>%  str_to_title())+
  theme_bw()+
  theme(legend.position = "none")+
  labs(x="Years to debut", y="Absolute Attribution")+
  coord_cartesian(xlim = c(1.2, 6.1))


# Sex Chapters ------------------------------------------------------------

chapter_attributions_sex <- positive_attributions_patients %>% 
  group_by(description, chapter, modality, sex) %>% 
  summarise(patient_average_attribution = mean(attribution)) %>% 
  mutate(abs_attribution=abs(patient_average_attribution),
         modality=case_match(modality,
                             "diag" ~ "diagnosis",
                             "ydelse" ~ "service",
                             .default=modality)) %>% 
  group_by(chapter, modality, sex) %>% 
  summarise(attribution = sum(abs_attribution)) %>% 
  ungroup()

chapter_attributions_sex_diff <- chapter_attributions_sex %>% 
  mutate(sex = if_else(sex ==1, 'Male', 'Female') )%>% 
  pivot_wider(id_cols=chapter,names_from=sex, values_from=attribution) %>% 
  mutate(difference=Male-Female,
         relative_difference = difference / Male)
 

chapter_attributions_sex_diff %>%
  mutate(chapter_label = if_else(chapter %in% top_25_chapters$chapter, chapter, "Other"),
         ordering = if_else(chapter_label != "Other", rank(relative_difference, ties.method = 'min'), 0)) %>%
  group_by(chapter_label,ordering) %>% 
  summarise(relative_difference=sum(relative_difference)) %>% 
  ggplot(aes(x=relative_difference, y=fct_reorder(chapter_label %>% str_to_title(), ordering), color=chapter_label))+
  geom_point()+
  geom_vline(aes(xintercept=0, alpha=0.2), linetype="dashed")+
  scale_color_manual(values=chapter_to_modality_color)+
  theme_bw()+
  theme(legend.position = "none")+
  labs(x="Difference in Aboslute Attributions", y="")



# Age Chapters ------------------------------------------------------------
chapter_attributions_age <- positive_attributions_patients %>% 
  mutate(age = year(outcome_date - days_to_censor) - year(birthdate),
         age_bracket = case_when(
           age <= 10 ~ "0-10",
           age > 10 & age <= 20 ~ "10-20",
           age > 20 & age <= 30 ~ "20-30",
           age > 30 & age <= 40 ~ "30-40",
           age > 40 ~ "40+",
           
         )) %>% 
  group_by(description, chapter, modality, age_bracket) %>% 
  summarise(patient_average_attribution = mean(attribution)) %>% 
  mutate(abs_attribution=abs(patient_average_attribution),
         modality=case_match(modality,
                             "diag" ~ "diagnosis",
                             "ydelse" ~ "service",
                             .default=modality)) %>% 
  group_by(chapter, modality, age_bracket) %>% 
  summarise(attribution = sum(abs_attribution)) %>% 
  ungroup()

chapter_age_diff <- chapter_attributions_age %>% 
  pivot_wider(values_from=attribution, names_from=age_bracket) %>% 
  mutate(diff=`30-40` - `10-20`)

diff_age_break <- sort(chapter_age_diff$diff,decreasing = TRUE)[15]


chapter_attributions_age %>%
  filter(chapter %in% top_25_chapters$chapter) %>% 
  group_by(chapter, age_bracket) %>% 
  summarise(attribution=sum(attribution)) %>% 
  ungroup() %>% 
  inner_join(chapter_age_diff %>% select(chapter, diff), by="chapter") %>% 
  mutate(chapter_label=if_else(age_bracket == "30-40" & diff >= diff_age_break &  chapter %in% top_25_chapters$chapter, chapter, ""),
         alpha = if_else(diff >= diff_age_break, 1, 0.5)) %>% 
  ggplot(aes(x=age_bracket, y=attribution, color=chapter, group=chapter, alpha=alpha))+
  geom_line(linewidth=1.5)+
  geom_point(size=3)+
  geom_text_repel(aes(label=chapter_label %>% str_to_title()), hjust=0, nudge_x = 0.1, direction = "y")+
  scale_color_manual(values=chapter_to_modality_color)+
  theme_bw()+
  theme(legend.position = "none")+
  labs(x="Age", y="Absolute Attribution")+
  coord_cartesian(xlim = c(.2, 5))



# Top Codes total ---------------------------------------------------------
codes_per_modality <- 15

features <- read_tsv("data/diabetes/features/features_to_filter.tsv")

top_codes <- code_attributions %>% 
  group_by(modality) %>%
  distinct(description, chapter, .keep_all = TRUE) %>% 
  slice_max(order_by = attribution, n=codes_per_modality) %>% 
  left_join(features, by=join_by(code == token)) %>% 
  mutate(label= as_factor(str_c(description.y, " (", code, ")", sep = "")) %>% fct_reorder(abs(attribution), .fun=max))

# export codes for interaction analysis
top_codes %>% select(code, modality) %>% write_delim("data/diabetes/attributions/interaction_codes.tsv", delim = "\t", col_names = FALSE)



top_codes_patients <- positive_attributions_patients %>%
  mutate(patient_type="Future Diabetes Debut") %>% 
  bind_rows(negative_attributions_patients %>%  mutate(patient_type="No Future Debut")) %>% 
  left_join(features %>% select(token, description), by=join_by(code == token)) %>% 
  mutate(code = if_else(modality == "diag", str_sub(code,2), code)) %>% 
  filter(description.y %in% top_codes$description.y) %>%
  mutate(
    modality=case_match(modality,
                        "diag" ~ "diagnosis",
                        "ydelse" ~ "service",
                        .default=modality),
    label = factor(str_c(description.y, " (", code, ")", sep = "")) %>% 
      fct_relevel(rev(levels(top_codes$label)))
    )


  
plots <- map(c('diagnosis', 'prescription', 'service'),
      \(mod){
        top_codes_patients %>%
          filter(modality == mod) %>% 
          ggplot(mapping = aes(x = attribution, y=label, fill=patient_type))+
          geom_boxplot(outlier.size = 0.1, size=0.3)+
          #scale_fill_manual()+
          scale_y_discrete(limits=rev)+
          scale_alpha(guide = "none") +
          theme_bw()+
          labs(fill="", y="", x="Attribution")+
          xlim(c(-1.5,2 ))
      })
plots[[1]]  / plots[[2]] / plots[[3]] + plot_layout(guides = "collect")
    
plots[[1]]
plots[[2]]
plots[[3]]



# Top codes per year ------------------------------------------------------

negative_levels <- negative_attributions_patients %>% 
  filter(str_c(chapter, description, sep = " - ") %in% top_codes$label) %>% 
  mutate(years_to_diagnosis="Negative",
         modality=case_match(modality,
                             "diag" ~ "diagnosis",
                             "ydelse" ~ "service",
                             .default=modality)
  )

positive_yearly_and_negative <- positive_attributions_patients_yearly %>% 
  mutate(years_to_diagnosis = years_to_diagnosis %>% as.character(),
         modality=case_match(modality,
                             "diag" ~ "diagnosis",
                             "ydelse" ~ "service",
                             .default=modality)) %>% 
  filter(str_c(chapter, description, sep = " - ") %in% top_codes$label) %>% 
  bind_rows(negative_levels)


map(c('diagnosis', 'prescription', 'service'),
      \(mod){
      positive_yearly_and_negative %>% 
      filter(modality == mod) %>% 
      mutate(alpha = if_else(years_to_diagnosis == "Negative", 0.67, 1),
             label = factor(str_c(chapter,description, sep=" - ")) %>% 
               fct_relevel(rev(levels(top_codes$label)))
             ) %>% 
      ggplot(aes(x=factor(years_to_diagnosis), y=attribution, fill=modality))+
        geom_violin(aes(alpha = alpha), draw_quantiles = 0.5)+
        scale_fill_manual(values=color_mapping_modality)+
        scale_alpha(guide = "none") +
        facet_wrap(vars(label), scales = "free_y")+
        theme_bw()+
        labs(fill="Modality", y="Feature importance", x="Years to diagnosis")+
        theme(legend.position = "none")
})


