library(readr)
library(dplyr)
library(rstatix)

# Load the dataframe exported from Python
dat <- read_tsv("anova_4way_unpooled_balanced_data.tsv")

# Make sure categorical variables are factors
dat <- dat |>
  mutate(
    session_uuid = factor(session_uuid),
    modulation = factor(modulation),
    feature = factor(feature),
    source = factor(source),
    rating_stimulus = factor(rating_stimulus)
  )

# 4-way repeated-measures ANOVA
anova_results <- dat |>
  anova_test(
    dv = mean_rating,
    wid = session_uuid,
    within = c(
      modulation,
      feature,
      source,
      rating_stimulus
    ),
    effect.size = "ges"
  )

get_anova_table(anova_results, correction = "none")
get_anova_table(anova_results, correction = "GG")
get_anova_table(anova_results, correction = "HF")
