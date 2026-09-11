#install.packages("dplyr")
#install.packages("tidyr")
#install.packages("rstatix")
#install.packages("emmeans")
#install.packages("ggh4x") 
#install.packages("ggsignif")
library(ggh4x)
library(dplyr)
library(tidyr)
library(rstatix)
library(ggplot2)
library(purrr)
library(broom)
library(emmeans)
library(ggsignif)

mushra = read.csv("~/Downloads/mushra.csv")
n_distinct(mushra$session_uuid)
# -----------------------------
# 1. Basic setup
# -----------------------------

data_clean <- mushra %>%
  filter(
    trial_id != "training",
    is.na(comments) | tolower(trimws(comments)) != "christhetree"
  )

# -----------------------------
# 2. Filter unsuitable devices
# -----------------------------

bad_devices <- c(
  "phone speakers",
  "laptop speakers",
  "computer speakers",
  "speaker",
  "speakers"
)

data_clean <- data_clean %>%
  filter(!tolower(listening_device) %in% bad_devices)



bad_trials <- data_clean %>%
  group_by(session_uuid, trial_id) %>%
  summarise(
    reference_score = rating_score[rating_stimulus == "reference"][1],
    all_identical = n_distinct(rating_score) == 1,
    total_time = max(rating_time, na.rm = TRUE),
    rating_range = max(rating_score, na.rm = TRUE) - min(rating_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(
    reference_score > 25 |
      all_identical |
      total_time < 24000 |
      rating_range < 10
  )



data_filtered <- data_clean %>%
  anti_join(
    bad_trials,
    by = c("session_uuid", "trial_id")
  )

# -----------------------------
# 5. Keep only non-reference ratings
# -----------------------------

mod_data <- data_filtered %>%
  filter(rating_stimulus != "reference")

t.test(dataPath$rating_score, mu = 0, alternative = "greater")


mod_data <- mod_data %>%
  separate(trial_id,
           into=c("modulation","feature","source"),
           sep="_",
           remove = FALSE)
mod_data %>% summarise(n_participants = n_distinct(session_uuid))

mod_avg <- mod_data %>%
  group_by(session_uuid, trial_id, modulation, feature, source) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )
mod_avg %>%
  count(session_uuid) %>%
  count(n)

anova_results <- mod_avg %>%
  anova_test(
    dv = mean_rating,
    wid = session_uuid,
    within = c(modulation, feature, source)
  )

anova_results

mod_avg_2way <- mod_data %>%
  filter(rating_stimulus != "reference") %>%
  group_by(
    session_uuid,
    modulation,
    feature
  ) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )

# 2-way repeated-measures ANOVA
anova_results_2way <- mod_avg_2way %>%
  anova_test(
    dv = mean_rating,
    wid = session_uuid,
    within = c(modulation, feature)
  )

anova_results_2way

model <- aov(
  mean_rating ~ modulation * feature +
    Error(session_uuid /
            (modulation * feature)),
  data = mod_avg_2way
)
emmeans(model, ~ modulation)
plot_data <- data_filtered %>%
  separate(trial_id,
           into = c("modulation", "feature", "source"),
           sep = "_",
           remove = FALSE) %>%
  group_by(trial_id) %>%
  mutate(
    modulation_amount = ifelse(
      rating_stimulus == "reference",
      0,
      as.numeric(factor(
        rating_stimulus[rating_stimulus != "reference"]
      ))[match(rating_stimulus,
               rating_stimulus[rating_stimulus != "reference"])]
    )
  ) %>%
  ungroup()
plot_data <- plot_data %>%
  mutate(
    # Make the five modulation levels categorical
    amount_plot = factor(
      modulation_amount,
      levels = sort(unique(modulation_amount)),
      labels = 1:5
    ),
    
    # Publication-friendly facet labels
    modulation = recode(
      modulation,
      amp  = "Amplitude",
      freq = "Frequency (Hz)",
      reg  = "Irregularity (%)"
    ),
    
    feature = recode(
      feature,
      brightness = "Brightness",
      richness   = "Richness",
      warmth     = "Warmth"
    ),
    
    source = recode(
      source,
      real      = "Real",
      synthetic = "Synthetic"
    )
  )


# --------------------------------------------------
# Plot
# --------------------------------------------------

ggplot(
  plot_data,
  aes(
    x = amount_plot,
    y = rating_score
  )
) +
  
  geom_boxplot(
    width = 0.48,
    alpha = 0.20,
    linewidth = 0.45,
    outlier.alpha = 0.30,
    outlier.size = 1.1
  ) +
  
  # Linear relationship
  geom_smooth(
    aes(x = as.numeric(amount_plot)),
    method = "lm",
    se = TRUE,
    colour = "#3366FF",
    fill = "#3366FF",
    alpha = 0.10,
    linewidth = 0.8
  ) +
  
  # 18 panels
  facet_grid(
    feature + source ~ modulation,
    scales = "free_x"
  ) +
  
  # Correct labels for each modulation type
  ggh4x::facetted_pos_scales(
    x = list(
      
      modulation == "Amplitude" ~
        scale_x_discrete(
          labels = c(
            "0.1",
            "0.3",
            "0.5",
            "0.7",
            "0.9"
          )
        ),
      
      modulation == "Frequency (Hz)" ~
        scale_x_discrete(
          labels = c(
            "0.25",
            "0.50",
            "1.00",
            "2.00",
            "4.00"
          )
        ),
      
      modulation == "Irregularity (%)" ~
        scale_x_discrete(
          labels = c(
            "0",
            "12.5",
            "25.0",
            "37.5",
            "50"
          )
        )
    )
  ) +
  
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    expand = expansion(mult = c(0.02, 0.03))
  ) +
  
  labs(
    x = "Modulation amount",
    y = "Perceived difference rating"
  ) +
  
  theme_bw(base_size = 11) +
  
  theme(
    
    # No title
    plot.title = element_blank(),
    
    # Axis titles
    axis.title.x = element_text(
      face = "bold",
      size = 12,
      margin = margin(t = 8)
    ),
    
    axis.title.y = element_text(
      face = "bold",
      size = 12,
      margin = margin(r = 8)
    ),
    
    # Axis numbers
    axis.text.x = element_text(
      size = 8.5,
      colour = "black"
    ),
    
    axis.text.y = element_text(
      size = 8.5,
      colour = "black"
    ),
    
    # Facet headings
    strip.text = element_text(
      face = "bold",
      size = 9.5,
      colour = "black"
    ),
    
    strip.background = element_rect(
      fill = "grey95",
      colour = "grey65",
      linewidth = 0.4
    ),
    
    # Subtle grid
    panel.grid.major = element_line(
      colour = "grey90",
      linewidth = 0.35
    ),
    
    panel.grid.minor = element_blank(),
    
    # Subtle panel boundaries
    panel.border = element_rect(
      colour = "grey60",
      linewidth = 0.4
    ),
    
    # Space between panels
    panel.spacing = unit(0.10, "cm"),
    
    # Clean margins
    plot.margin = margin(5, 8, 5, 5)
  )

n_distinct(plot_data$session_uuid)
plot_data_combined <- plot_data_combined %>%
  mutate(
    modulation_label = case_when(
      
      # Amplitude
      modulation == "amp" & modulation_amount == 0 ~ "0.1",
      modulation == "amp" & modulation_amount == 1 ~ "0.3",
      modulation == "amp" & modulation_amount == 2 ~ "0.5",
      modulation == "amp" & modulation_amount == 3 ~ "0.7",
      modulation == "amp" & modulation_amount == 4 ~ "0.9",
      
      # Frequency
      modulation == "freq" & modulation_amount == 0 ~ "0.25",
      modulation == "freq" & modulation_amount == 1 ~ "0.5",
      modulation == "freq" & modulation_amount == 2 ~ "1",
      modulation == "freq" & modulation_amount == 3 ~ "2",
      modulation == "freq" & modulation_amount == 4 ~ "4",
      
      # Irregularity
      modulation == "reg" & modulation_amount == 0 ~ "0",
      modulation == "reg" & modulation_amount == 1 ~ "12.5",
      modulation == "reg" & modulation_amount == 2 ~ "25",
      modulation == "reg" & modulation_amount == 3 ~ "37.5",
      modulation == "reg" & modulation_amount == 4 ~ "50"
    )
  )


# ---------------------------------------------------------
# Participant-level means
# ---------------------------------------------------------

confidence_data <- plot_data_combined %>%
  group_by(
    session_uuid,
    modulation,
    modulation_amount,
    modulation_label
  ) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )


# ---------------------------------------------------------
# Group means + 95% confidence intervals
# ---------------------------------------------------------

summary_data <- confidence_data %>%
  group_by(
    modulation,
    modulation_amount,
    modulation_label
  ) %>%
  summarise(
    mean = mean(mean_rating, na.rm = TRUE),
    ci = qt(0.975, n() - 1) *
      sd(mean_rating, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )


# ---------------------------------------------------------
# R-squared for each modulation type
# ---------------------------------------------------------

r2_data <- summary_data %>%
  group_by(modulation) %>%
  summarise(
    r2 = summary(
      lm(mean ~ modulation_amount)
    )$r.squared,
    .groups = "drop"
  ) %>%
  mutate(
    r2_label = paste0("R² = ", sprintf("%.2f", r2))
  )
plot_data <- plot_data %>%
  mutate(
    modulation = case_when(
      modulation == "amp"  ~ "Amplitude",
      modulation == "freq" ~ "Frequency",
      modulation == "reg"  ~ "Irregularity",
      TRUE ~ as.character(modulation)
    ),
    modulation = factor(
      modulation,
      levels = c("Amplitude", "Frequency", "Irregularity")
    )
  )
summary_sd <- plot_data_combined %>%
  group_by(
    modulation,
    modulation_amount,
    modulation_label
  ) %>%
  summarise(
    mean = mean(rating_score, na.rm = TRUE),
    sd = sd(rating_score, na.rm = TRUE),
    .groups = "drop"
  )

r2_data <- summary_sd %>%
  group_by(modulation) %>%
  summarise(
    r2 = summary(
      lm(mean ~ modulation_amount)
    )$r.squared,
    .groups = "drop"
  ) %>%
  mutate(
    r2_label = paste0(
      "R² = ",
      sprintf("%.2f", r2)
    )
  )

r2_data
# ---------------------------------------------------------
# Figure 1
# ---------------------------------------------------------
modulation_amount_sd_plot <- ggplot(
  summary_sd,
  aes(
    x = modulation_amount,
    y = mean
  )
) +
  
  # Standard deviation
  geom_errorbar(
    aes(
      ymin = mean - sd,
      ymax = mean + sd
    ),
    width = 0.10,
    linewidth = 0.6
  ) +
  
  # Circular points for amounts 1–4
  geom_point(
    data = summary_sd %>%
      filter(modulation_amount != 0),
    size = 2.6
  ) +
  
  # Square for minimally modulated reference
  geom_point(
    data = summary_sd %>%
      filter(modulation_amount == 0),
    shape = 15,
    size = 2.8
  ) +
  
  # Blue linear regression
  geom_smooth(
    method = "lm",
    se = FALSE,
    linewidth = 1,
    colour = "blue"
  ) +
  
  # R-squared annotation
  geom_text(
    data = r2_data,
    aes(
      x = -Inf,
      y = Inf,
      label = r2_label
    ),
    inherit.aes = FALSE,
    hjust = -0.12,
    vjust = 1.4,
    size = 5.2,
    fontface = "bold",
    colour = "blue"
  ) +
  
  facet_wrap(
    ~ modulation,
    nrow = 1,
    scales = "free_x",
    labeller = as_labeller(
      c(
        amp = "Amplitude",
        freq = "Frequency (Hz)",
        reg = "Irregularity (%)"
      )
    )
  ) +
  
  ggh4x::facetted_pos_scales(
    x = list(
      
      modulation == "amp" ~
        scale_x_continuous(
          breaks = 0:4,
          labels = c("0.1", "0.3", "0.5", "0.7", "0.9"),
          expand = expansion(mult = c(0.06, 0.06))
        ),
      
      modulation == "freq" ~
        scale_x_continuous(
          breaks = 0:4,
          labels = c("0.25", "0.5", "1", "2", "4"),
          expand = expansion(mult = c(0.06, 0.06))
        ),
      
      modulation == "reg" ~
        scale_x_continuous(
          breaks = 0:4,
          labels = c("0", "12.5", "25", "37.5", "50"),
          expand = expansion(mult = c(0.06, 0.06))
        )
    )
  ) +
  
  scale_y_continuous(
    breaks = seq(0, 100, 25),
    expand = expansion(mult = c(0, 0))
  ) +
  
  coord_cartesian(
    ylim = c(0, 110)
  ) +
  
  labs(
    x = "Modulation amount",
    y = "Perceived difference rating"
  ) +
  
  theme_bw(base_size = 15) +
  
  theme(
    panel.border = element_blank(),
    panel.grid.minor = element_blank(),
    
    strip.background = element_rect(
      fill = "grey90",
      colour = "black"
    ),
    
    strip.text = element_text(
      size = 17,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      size = 13,
      margin = margin(t = 5)
    ),
    
    axis.text.y = element_text(
      size = 14,
      margin = margin(r = 5)
    ),
    
    axis.title.x = element_text(
      size = 18,
      face = "bold",
      margin = margin(t = 10)
    ),
    
    axis.title.y = element_text(
      size = 18,
      face = "bold",
      margin = margin(r = 12)
    ),
    
    panel.spacing = grid::unit(0.8, "lines"),
    aspect.ratio = 2,
    plot.background = element_blank(),
    
    plot.margin = margin(
      t = 5,
      r = 5,
      b = 5,
      l = 5
    )
  )

modulation_amount_sd_plot

modulation_amount_boxplot <- ggplot(
  plot_data_combined,
  aes(
    x = modulation_amount,
    y = rating_score
  )
) +
  
  # Blue linear regression line
  geom_smooth(
    method = "lm",
    se = FALSE,
    linewidth = 1,
    colour = "blue"
  ) +
  
  # Boxplots
  geom_boxplot(
    aes(group = modulation_amount),
    width = 0.45,
    alpha = 0.3,
    outlier.shape = NA,
    linewidth = 0.6
  ) +
  
  # R-squared annotation
  geom_text(
    data = r2_data,
    aes(
      x = -Inf,
      y = Inf,
      label = r2_label
    ),
    inherit.aes = FALSE,
    hjust = -0.12,
    vjust = 1.4,
    size = 5.2,
    fontface = "bold",
    colour = "blue"
  ) +
  
  facet_wrap(
    ~ modulation,
    nrow = 1,
    scales = "free_x",
    labeller = as_labeller(
      c(
        amp = "Amplitude",
        freq = "Frequency (Hz)",
        reg = "Irregularity (%)"
      )
    )
  ) +
  
  ggh4x::facetted_pos_scales(
    x = list(
      
      modulation == "amp" ~
        scale_x_continuous(
          breaks = 0:4,
          labels = c("0.1", "0.3", "0.5", "0.7", "0.9"),
          expand = expansion(mult = c(0.06, 0.06))
        ),
      
      modulation == "freq" ~
        scale_x_continuous(
          breaks = 0:4,
          labels = c("0.25", "0.5", "1", "2", "4"),
          expand = expansion(mult = c(0.06, 0.06))
        ),
      
      modulation == "reg" ~
        scale_x_continuous(
          breaks = 0:4,
          labels = c("0", "12.5", "25", "37.5", "50"),
          expand = expansion(mult = c(0.06, 0.06))
        )
    )
  ) +
  
  scale_y_continuous(
    breaks = seq(0, 100, 25),
    expand = expansion(mult = c(0, 0))
  ) +
  
  coord_cartesian(
    ylim = c(0, 103)
  ) +
  
  labs(
    x = "Modulation amount",
    y = "Perceived difference rating"
  ) +
  
  theme_bw(base_size = 15) +
  
  theme(
    panel.border = element_blank(),
    panel.grid.minor = element_blank(),
    
    strip.background = element_rect(
      fill = "grey90",
      colour = "black"
    ),
    
    strip.text = element_text(
      size = 17,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      size = 13,
      margin = margin(t = 5)
    ),
    
    axis.text.y = element_text(
      size = 14,
      margin = margin(r = 5)
    ),
    
    axis.title.x = element_text(
      size = 18,
      face = "bold",
      margin = margin(t = 10)
    ),
    
    axis.title.y = element_text(
      size = 18,
      face = "bold",
      margin = margin(r = 12)
    ),
    
    panel.spacing = grid::unit(0.8, "lines"),
    aspect.ratio = 2,
    plot.background = element_blank(),
    
    plot.margin = margin(
      t = 5,
      r = 5,
      b = 5,
      l = 5
    )
  )

modulation_amount_boxplot
ggsave(
  "modulation_amount_plot.png",
  modulation_amount_plot,
  width = 13,
  height = 4.5,
  units = "in",
  dpi = 300,
  bg = "white"
)

mod_avg_amount <- plot_data %>%
  group_by(
    session_uuid,
    modulation,
    feature,
    source,
    modulation_amount
  ) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )
normality_results_amount <- mod_avg_amount %>%
  group_by(
    modulation_amount,
    modulation,
    feature,
    source
  ) %>%
  summarise(
    n = sum(!is.na(mean_rating)),
    W = shapiro.test(mean_rating)$statistic,
    p = shapiro.test(mean_rating)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    normality = ifelse(
      p > 0.05,
      "Normal",
      "Non-normal"
    )
  )

print(normality_results_amount, n = 90)

plot_data_grouped <- plot_data %>%
  filter(modulation_amount %in% 1:4) %>%
  mutate(
    amount_group = case_when(
      modulation_amount %in% c(1, 2) ~ "Low",
      modulation_amount %in% c(3, 4) ~ "High"
    ),
    amount_group = factor(
      amount_group,
      levels = c("Low", "High")
    ),
    modulation = factor(
      modulation,
      levels = c("amp", "freq", "reg")
    )
  )

plot_data_grouped <- plot_data_grouped %>%
  mutate(
    modulation = recode(
      modulation,
      "amp" = "Amplitude",
      "freq" = "Frequency",
      "reg" = "Irregularity"
    ),
    feature = recode(
      feature,
      "brightness" = "Brightness",
      "warmth" = "Warmth",
      "richness" = "Richness"
    ),
    source = recode(
      source,
      "real" = "Natural",
      "synthetic" = "Synthetic"
    ),
    amount_group = factor(
      amount_group,
      levels = c("Low", "High")
    )
  )
mod_avg_2way <- plot_data_grouped %>%
  group_by(
    session_uuid,
    modulation,
    amount_group
  ) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )

anova_results_2way <- mod_avg_2way %>%
  anova_test(
    dv = mean_rating,
    wid = session_uuid,
    within = c(modulation, amount_group)
  )

anova_results_2way

ggplot(
  plot_data_grouped,
  aes(
    x = amount_group,
    y = rating_score,
    fill = amount_group
  )
) +
  
  geom_boxplot(
    width = 0.50,
    alpha = 0.55,
    linewidth = 0.45,
    outlier.alpha = 0.25,
    outlier.size = 1
  ) +
  
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 2,
    fill = "white",
    stroke = 0.6
  ) +
  
  ggh4x::facet_grid2(
    feature + source ~ modulation,
    axes = "all"
  ) +
  
  scale_y_continuous(
    limits = c(0, 100),
    breaks = seq(0, 100, 25),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  
  labs(
    x = "Modulation amount",
    y = "Perceived difference rating"
  ) +
  theme(
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold")
  ) + 
  guides(fill = "none") +
  
  theme_classic(base_size = 11) +
  
  theme(
    strip.background = element_blank(),
    
    strip.text = element_text(
      face = "bold",
      size = 9
    ),
    
    axis.text.x = element_text(
      size = 8
    ),
    
    axis.text.y = element_text(
      size = 8
    ),
    
    axis.title = element_text(
      size = 11
    ),
    
    panel.spacing = unit(0.8, "lines")
  )
print(normality_low_high, n = 36)
# One mean per participant, modulation type, and amount group

anova_data_4way <- plot_data_grouped %>%
  group_by(
    session_uuid,
    modulation,
    feature,
    source,
    amount_group
  ) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )


anova_results_4way
r2_table <- plot_data %>%
  group_by(modulation, feature, source) %>%
  summarise(
    R2 = summary(lm(rating_score ~ modulation_amount))$r.squared,
    p = summary(lm(rating_score ~ modulation_amount))$coefficients[2,4],
    .groups = "drop"
  )

print(r2_table)
ggsave(
  "modulation_types_within_low_high.png",
  plot = modulation_by_amount,
  width = 7.5,
  height = 3.4,
  units = "in",
  dpi = 300
)
modulation_means <- mod_avg %>%
  group_by(session_uuid, modulation) %>%
  summarise(
    mean_rating = mean(mean_rating),
    .groups = "drop"
  )

complete_sessions <- modulation_means %>%
  group_by(session_uuid) %>%
  summarise(
    n_modulations = n_distinct(modulation),
    .groups = "drop"
  ) %>%
  filter(n_modulations == 3) %>%
  pull(session_uuid)

modulation_means_complete <- modulation_means %>%
  filter(session_uuid %in% complete_sessions)

pairwise_results <- modulation_means_complete %>%
  pairwise_t_test(
    mean_rating ~ modulation,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )

pairwise_results

feature_means <- mod_data %>%
  group_by(session_uuid, feature) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(!is.na(mean_rating))
feature_means_complete <- feature_means %>%
  pivot_wider(
    names_from = feature,
    values_from = mean_rating
  ) %>%
  drop_na(brightness, richness, warmth) %>%
  pivot_longer(
    cols = c(brightness, richness, warmth),
    names_to = "feature",
    values_to = "mean_rating"
  ) %>%
  arrange(session_uuid, feature)
feature_pairwise <- feature_means_complete %>%
  pairwise_t_test(
    mean_rating ~ feature,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )

feature_pairwise

linear_results <- plot_data %>%
  group_by(modulation, feature, source) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lm(
      rating_score ~ modulation_amount,
      data = .x
    )),
    r_squared = map_dbl(model, ~ summary(.x)$r.squared)
  ) %>%
  select(modulation, feature, source, r_squared) %>%
  mutate(model_type = "Linear")


# -----------------------------
# Exponential model
# y = a * exp(bx)
#
# Only uses ratings > 0 because log(0)
# is undefined
# -----------------------------

exponential_results <- plot_data %>%
  filter(rating_score > 0) %>%
  group_by(modulation, feature, source) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lm(
      log(rating_score) ~ modulation_amount,
      data = .x
    )),
    r_squared = map_dbl(model, ~ summary(.x)$r.squared)
  ) %>%
  select(modulation, feature, source, r_squared) %>%
  mutate(model_type = "Exponential")


# -----------------------------
# Combine results
# -----------------------------

model_comparison <- bind_rows(
  linear_results,
  exponential_results
)
mod_grouped <- plot_data %>%
  filter(modulation_amount %in% 1:4) %>%
  mutate(
    amount_group = case_when(
      modulation_amount %in% c(1, 2) ~ "low",
      modulation_amount %in% c(3, 4) ~ "high"
    ),
    amount_group = factor(
      amount_group,
      levels = c("low", "high")
    ),
    modulation = factor(modulation),
    feature = factor(feature),
    source = factor(source)
  )
mod_avg_grouped <- mod_grouped %>%
  group_by(
    session_uuid,
    modulation,
    feature,
    source,
    amount_group
  ) %>%
  summarise(
    mean_rating = mean(rating_score, na.rm = TRUE),
    .groups = "drop"
  )


normality_results <- mod_avg_grouped %>%
  group_by(amount_group, modulation, feature, source) %>%
  summarise(
    W = shapiro.test(mean_rating)$statistic,
    p = shapiro.test(mean_rating)$p.value,
    .groups = "drop"
  )

print(n = 36, normality_results)

print(n = 36, model_comparison)
# ---------------------------------------------------------
# Participant-level means for low/high modulation groups

## A lot of this data here is designed for ISMIR's Late
#Breaking Demo (LBD), you can ignore this for the dissertation.
# ---------------------------------------------------------
# 
# plot_data_lbd <- plot_data_grouped %>%
#   group_by(
#     session_uuid,
#     modulation,
#     amount_group
#   ) %>%
#   summarise(
#     mean_rating = mean(rating_score, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# ttest_low_high <- plot_data_lbd %>%
#   group_by(modulation) %>%
#   pairwise_t_test(
#     mean_rating ~ amount_group,
#     paired = TRUE,
#     p.adjust.method = "bonferroni"
#   )
# 
# ttest_low_high
# # ---------------------------------------------------------
# # Significance bars
# # Replace the stars with your actual statistical results
# # ---------------------------------------------------------
# 
# sig_low_high <- plot_data_lbd %>%
#   group_by(modulation) %>%
#   summarise(
#     max_value = max(mean_rating, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   mutate(
#     y_position = case_when(
#       modulation == "freq" ~ max_value + 7,
#       TRUE ~ max_value + 5
#     ),
#     xmin = 1,
#     xmax = 2,
#     annotation = "***"
#   )
# # ---------------------------------------------------------
# # Plot
# # ---------------------------------------------------------
# names(plot_data)
# plot_data <- plot_data %>%
#   mutate(
#     modulation = case_when(
#       grepl("^amp_", trial_id) ~ "Amplitude",
#       grepl("^freq_", trial_id) ~ "Frequency",
#       grepl("^reg_", trial_id) ~ "Irregularity",
#       TRUE ~ NA_character_
#     ),
#     
#     modulation = factor(
#       modulation,
#       levels = c(
#         "Amplitude",
#         "Frequency",
#         "Irregularity"
#       )
#     )
#   )
# plot_data_grouped <- plot_data %>%
#   filter(modulation_amount %in% 1:4) %>%
#   mutate(
#     amount_group = case_when(
#       modulation_amount %in% c(1, 2) ~ "Low",
#       modulation_amount %in% c(3, 4) ~ "High"
#     ),
#     
#     amount_group = factor(
#       amount_group,
#       levels = c("Low", "High")
#     )
#   )
# plot_data_lbd <- plot_data_grouped %>%
#   group_by(
#     session_uuid,
#     modulation,
#     amount_group
#   ) %>%
#   summarise(
#     mean_rating = mean(rating_score, na.rm = TRUE),
#     .groups = "drop"
#   )
# low_high_by_modulation <- ggplot(
#   plot_data_lbd,
#   aes(
#     x = amount_group,
#     y = mean_rating
#   )
# ) +
#   
#   geom_boxplot(
#     width = 0.55,
#     alpha = 0.30,
#     outlier.shape = NA,
#     linewidth = 0.7
#   ) +
#   
#   geom_jitter(
#     width = 0.08,
#     height = 0,
#     size = 1.8,
#     alpha = 0.55
#   ) +
#   
#   facet_wrap(
#     ~ modulation,
#     nrow = 1,
#     labeller = as_labeller(
#       c(
#         Amplitude = "Amplitude",
#         Frequency = "Frequency (Hz)",
#         Irregularity = "Irregularity (%)"
#       )
#     )
#   ) +
#   
#   scale_y_continuous(
#     breaks = seq(0, 100, 25),
#     expand = expansion(mult = c(0, 0))
#   ) +
#   
#   coord_cartesian(
#     ylim = c(0, 105)
#   ) +
#   
#   labs(
#     x = NULL,
#     y = "Perceived difference rating"
#   ) +
#   
#   theme_bw(base_size = 15) +
#   
#   theme(
#     legend.position = "none",
#     
#     panel.border = element_blank(),
#     panel.grid.minor = element_blank(),
#     
#     strip.background = element_rect(
#       fill = "grey90",
#       colour = "black",
#       linewidth = 0.5
#     ),
#     
#     strip.text = element_text(
#       size = 17,
#       face = "bold"
#     ),
#     
#     axis.text.x = element_text(
#       size = 14,
#       margin = margin(t = 5)
#     ),
#     
#     axis.text.y = element_text(
#       size = 14,
#       margin = margin(r = 5)
#     ),
#     
#     axis.title.y = element_text(
#       size = 18,
#       face = "bold",
#       margin = margin(r = 12)
#     ),
#     
#     panel.spacing = grid::unit(0.8, "lines"),
#     
#     aspect.ratio = 1,
#     
#     plot.background = element_blank(),
#     
#     plot.margin = margin(
#       t = 5,
#       r = 5,
#       b = 5,
#       l = 5
#     )
#   )
# 
# low_high_by_modulation
# 
# # -------------------------------------------------------
# # Significance brackets
# # -------------------------------------------------------
# 
# sig_inverted <- tibble(
#   amount_group = factor(
#     c(
#       "Low", "Low", "Low",
#       "High", "High", "High"
#     ),
#     levels = c("Low", "High")
#   ),
#   
#   xmin = c(
#     1, 2, 1,
#     1, 2, 1
#   ),
#   
#   xmax = c(
#     2, 3, 3,
#     2, 3, 3
#   ),
#   
#   y_position = c(
#     66, 74, 82,
#     100, 106, 112
#   ),
#   
#   annotation = c(
#     "***", "***", "***",
#     "***", "***", "***"
#   )
# )
# 
# 
# # -------------------------------------------------------
# # Plot
# # -------------------------------------------------------
# 
# plot_data_inverted <- plot_data_combined %>%
#   
#   filter(modulation_amount %in% 1:4) %>%
#   
#   mutate(
#     amount_group = case_when(
#       modulation_amount %in% c(1, 2) ~ "Low",
#       modulation_amount %in% c(3, 4) ~ "High"
#     )
#   ) %>%
#   
#   # Average the two modulation amounts within each group
#   # for each participant and modulation type
#   group_by(
#     session_uuid,
#     modulation,
#     amount_group
#   ) %>%
#   
#   summarise(
#     mean_rating = mean(rating_score, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   
#   mutate(
#     modulation = factor(
#       modulation,
#       levels = c("amp", "freq", "reg"),
#       labels = c(
#         "Amplitude",
#         "Frequency",
#         "Irregularity"
#       )
#     ),
#     
#     amount_group = factor(
#       amount_group,
#       levels = c("Low", "High")
#     )
#   )
# 
# paired_modulation_data <- plot_data_inverted %>%
#   select(
#     session_uuid,
#     amount_group,
#     modulation,
#     mean_rating
#   ) %>%
#   pivot_wider(
#     names_from = modulation,
#     values_from = mean_rating
#   )
# 
# paired_modulation_data
# ttest_modulation <- paired_modulation_data %>%
#   group_by(amount_group) %>%
#   summarise(
#     
#     amp_vs_freq_p = t.test(
#       Amplitude,
#       Frequency,
#       paired = TRUE
#     )$p.value,
#     
#     amp_vs_reg_p = t.test(
#       Amplitude,
#       Irregularity,
#       paired = TRUE
#     )$p.value,
#     
#     freq_vs_reg_p = t.test(
#       Frequency,
#       Irregularity,
#       paired = TRUE
#     )$p.value,
#     
#     .groups = "drop"
#   )
# 
# ttest_modulation
# ---------------------------------------------------------
# Significance annotations
#
# 1 = Amplitude
# 2 = Frequency
# 3 = Periodicity
#
#FOR LBD
# ---------------------------------------------------------
# 
# sig_inverted <- tibble(
#   amount_group = factor(
#     c(
#       "Low", "Low", "Low",
#       "High", "High", "High"
#     ),
#     levels = c("Low", "High")
#   ),
#   
#   xmin = c(
#     1, 1, 2,
#     1, 1, 2
#   ),
#   
#   xmax = c(
#     2, 3, 3,
#     2, 3, 3
#   ),
#   
#   # Staggered vertically so brackets do not overlap
#   y_position = c(
#     66, 82, 74,
#     100, 112, 106
#   ),
#   
#   annotation = c(
#     "***", "***", "***",
#     "***", "***", "***"
#   )
# )
# 
# 
# # ---------------------------------------------------------
# # Plot
# # ---------------------------------------------------------
# 
# inverted_plot <- ggplot(
#   plot_data_inverted,
#   aes(
#     x = modulation,
#     y = mean_rating
#   )
# ) +
#   
#   geom_boxplot(
#     width = 0.55,
#     alpha = 0.30,
#     outlier.shape = NA,
#     linewidth = 0.7
#   ) +
#   
#   geom_jitter(
#     width = 0.08,
#     height = 0,
#     size = 1.8,
#     alpha = 0.55
#   ) +
#   
#   # -------------------------------------------------------
# # Statistical significance brackets
# # -------------------------------------------------------
# 
# geom_signif(
#   data = sig_inverted,
#   aes(
#     xmin = xmin,
#     xmax = xmax,
#     annotations = annotation,
#     y_position = y_position
#   ),
#   manual = TRUE,
#   inherit.aes = FALSE,
#   
#   tip_length = 0.025,
#   textsize = 5.5,
#   linewidth = 0.7,
#   vjust = 0.2
# ) +
#   
#   # -------------------------------------------------------
# # Low and High panels
# # -------------------------------------------------------
# 
# facet_wrap(
#   ~ amount_group,
#   nrow = 1
# ) +
#   
#   # -------------------------------------------------------
# # Y axis
# # Extra space above 100 is only for significance bars
# # -------------------------------------------------------
# 
# scale_y_continuous(
#   limits = c(0, 118),
#   breaks = seq(0, 100, 25),
#   expand = expansion(mult = c(0, 0))
# ) +
#   
#   labs(
#     x = "Modulation type",
#     y = "Perceived difference rating"
#   ) +
#   
#   # -------------------------------------------------------
# # Styling
# # -------------------------------------------------------
# 
# theme_bw(base_size = 15) +
#   
#   theme(
#     legend.position = "none",
#     
#     panel.border = element_blank(),
#     panel.grid.minor = element_blank(),
#     
#     strip.background = element_rect(
#       fill = "grey90",
#       colour = "black",
#       linewidth = 0.5
#     ),
#     
#     strip.text = element_text(
#       size = 17,
#       face = "bold"
#     ),
#     
#     axis.text.x = element_text(
#       size = 14,
#       margin = margin(t = 5)
#     ),
#     
#     axis.text.y = element_text(
#       size = 14,
#       margin = margin(r = 5)
#     ),
#     
#     axis.title.x = element_text(
#       size = 18,
#       face = "bold",
#       margin = margin(t = 10)
#     ),
#     
#     axis.title.y = element_text(
#       size = 18,
#       face = "bold",
#       margin = margin(r = 12)
#     ),
#     
#     panel.spacing = grid::unit(
#       0.8,
#       "lines"
#     ),
#     
#     aspect.ratio = 1,
#     
#     plot.background = element_blank(),
#     
#     plot.margin = margin(
#       t = 5,
#       r = 5,
#       b = 5,
#       l = 5
#     )
#   )
# 
# 
# inverted_plot
