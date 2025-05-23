library(tidyverse)
library(scales)
library(ggrepel)

color_mapping_modality <- list(
  Diagnosis = "firebrick",   
  Prescription = "#FFA500",
  Service = "cornflowerblue",
 `Diagnosis Service Prescription` = "palegreen4"
)

color_mapping_sex <- list(
  Female = "steelblue",   
  Male = "darkorange"
)

color_mapping_complication<- list(
  Complication = "chocolate2",   
  `No Complication` = "darkgreen"
)

files_to_gather = c(
  'summaries/diabetes/2024-07-24-transformer0/all_bugfix/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-30-logisitic0/all/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-24-transformer0/validation_quantile01/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-24-transformer0/validation_precision5/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-24-transformer0/sex_1/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-24-transformer0/sex_2/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-24-transformer0/complication_0/bootstrap_TableS4.Performance_table.csv',
  'summaries/diabetes/2024-07-24-transformer0/complication_1/bootstrap_TableS4.Performance_table.csv'
  
  
)

load_df <- tibble(
  files_to_gather = files_to_gather,
  run_type = c('Transformer', 'Bag Of Words', "Validation", "Validation_precision5", "Male", "Female", "No Complication", "Complication")
)

data <-load_df %>%  
  mutate(results = map(files_to_gather, read_csv)) %>% 
  unnest(results) %>% 
  mutate(Modality = str_replace_all(Modality,"_", " ")) %>% 
  mutate(Modality = str_replace_all(Modality, "diag", 'diagnosis'),
         Modality = str_replace_all(Modality, 'ydelse', 'service'),
         Modality = str_replace_all(Modality, 'prescription', 'prescription'),
         Modality = str_to_title(Modality) %>% as_factor() %>% fct_relevel("Diagnosis Service Prescription", "Service", "Prescription", "Diagnosis"))%>% 
  mutate(`Exclusion Interval`= `Prediction Interval` - 12) %>% 
  mutate(interval = str_c(`Exclusion Interval`, "-",`Prediction Interval`))


# Plotting ----------------------------------------------------------------


metric <- 'relative_risk'
df_selected_metric <- data %>% 
  dplyr::filter(Metric == metric) 

y_label <- function(metric){
  if (metric == 'relative_risk'){
    return ('Relative Risk (1000 per million at risk)')
  }
  if (metric == 'auroc'){
    return ('AUROC')
  }
  return(metric)
}
ylim_low <- function(metric){
  if (metric == 'auroc'){
    return (0.5)
    
  }
  return (0)
}

data_metric <- df_selected_metric %>% dplyr::filter(str_to_lower(run_type) %in% c("transformer", "bag of words") )

data_metric %>%
  filter(Modality == "Diagnosis Service Prescription") %>% 
  ggplot(mapping = aes(x=as_factor(interval), y=Median,  fill=run_type))+
  geom_bar(stat = 'identity', position = 'dodge')+
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high), position = 'dodge')+
  coord_cartesian(ylim=c(ylim_low(metric),NA)) +
  labs(y=y_label(metric),
       x='Assessment Interval',
       fill='Model')+
  theme_minimal()+
  theme(legend.position = 'bottom', text = element_text(size=15))


data_metric %>%
  filter(run_type == "Transformer") %>% 
  ggplot(mapping = aes(x=as_factor(interval), y=Median, fill=Modality))+
  geom_bar(stat = 'identity', position = 'dodge')+
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high), position = 'dodge')+
  facet_grid( rows=vars(run_type))+
  coord_cartesian(ylim=c(ylim_low(metric),NA)) +
  labs(y=y_label(metric),
       x='Assessment Interval',
       fill='Registry')+
  scale_fill_manual(values=color_mapping_modality)+
  theme_minimal()+
  theme(legend.position = 'bottom', text = element_text(size=15))



# Validation --------------------------------------------------------------
region_code_to_name = list(
  "1084"= "Capital Region Of Denmark",
  "1085"= "Region Zealand",
  "1083"= "Region of Southern Denmark",
  "1082"= "Central Denmark Region",
  "1081"= "North Denmark Region"
)

region_color_map = list(
  "Capital Region Of Denmark"= "#585b70",
  "Region Zealand"= "#c6a0f6",
  "Region of Southern Denmark"= "#f8bd96",
  "Central Denmark Region"= "#a6e3a1",
  "North Denmark Region"= "#89dceb",
  "NA"= "#b4befe"
)

validation <- data %>% 
  filter(str_starts(run_type,"Validation")) %>%
  mutate(region_code = str_extract(experiment_dir, '[:digit:]+$'),
         region = map_chr(region_code, ~region_code_to_name[[.x]]))
  

validation %>%
  filter(run_type == "Validation", Metric == "relative_risk") %>% 
  ggplot(mapping = aes(x=as_factor(interval), y=Median, color=region))+
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high), position = 'dodge')+
  coord_cartesian(ylim=c(ylim_low(metric),NA)) +
  labs(y=y_label(metric),
       x='Assessment Interval',
       color='Region')+
  scale_color_manual(values=region_color_map)+
  theme_minimal()+
  theme(legend.position = 'bottom', text = element_text(size=15))

validation %>%
  filter(run_type == "Validation_precision5", Metric=="recall") %>% 
  ggplot(mapping = aes(x=as_factor(interval), y=Median, color=region))+
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high), position = 'dodge')+
  coord_cartesian(ylim=c(ylim_low(metric),NA)) +
  labs(y=y_label("Recall"),
       x='Assessment Interval',
       color='Region')+
  scale_color_manual(values=region_color_map)+
  theme_minimal()+
  theme(legend.position = 'bottom', text = element_text(size=15))

# Fairness analysis -------------------------------------------------------

sex_fairness <- data %>% 
  filter(run_type %in% c("Male", "Female"))


sex_fairness %>%
  filter(Metric == "relative_risk", Modality == "Diagnosis Service Prescription") %>% 
  ggplot(mapping = aes(x=as_factor(interval), y=Median, color=run_type))+
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high))+
  coord_cartesian(ylim=c(ylim_low(metric),NA)) +
  labs(y=y_label("relative_risk"),
       x='Assessment Interval',
       color='Sex')+
  scale_color_manual(values=color_mapping_sex)+
  theme_minimal()+
  theme(legend.position = 'bottom', text = element_text(size=15))


# Complication analysis -------------------------------------------------------

complication <- data %>% 
  filter(run_type %in% c("Complication", "No Complication"))


complication %>%
  filter(Metric == "relative_risk", Modality == "Diagnosis Service Prescription") %>% 
  ggplot(mapping = aes(x=as_factor(interval), y=Median, color=run_type))+
  geom_errorbar(aes(ymin=ci_low, ymax=ci_high))+
  coord_cartesian(ylim=c(ylim_low(metric),NA)) +
  labs(y=y_label("relative_risk"),
       x='Assessment Interval',
       color="")+
  scale_color_manual(values=color_mapping_complication)+
  theme_minimal()+
  theme(legend.position = 'bottom', text = element_text(size=15))

