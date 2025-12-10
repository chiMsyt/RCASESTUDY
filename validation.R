# ==========================================================
# STATISTICAL VALIDATION SCRIPT
# Data Source: Synthetic_FIES_NCR.csv (Prototype)
# Technologies: Tidyverse, GGPlot2
# ==========================================================

# 1. AUTO-INSTALL LIBRARIES (Safety Check)
packages <- c("tidyverse", "ggplot2", "dplyr")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)

library(tidyverse)
library(ggplot2)
library(dplyr)

# 2. LOAD SYNTHETIC DATA
filename <- "Synthetic_FIES_NCR.csv"

if (!file.exists(filename)) {
    stop("Error: Please run 'python dummy_generator.py' first!")
}

print(paste(">>> LOADING PROTOTYPE DATA:", filename))
df <- read_csv(filename, show_col_types = FALSE)

# 3. DATA TRANSFORMATION (Mapping Districts)
# Even though it's dummy data, we treat it professionally
df <- df %>%
    mutate(District_Label = case_when(
        W_PROV == 39 ~ "Manila",
        W_PROV == 74 ~ "Quezon City",
        W_PROV == 75 ~ "North NCR",
        W_PROV == 76 ~ "South/Makati",
        TRUE ~ "Other"
    ))

# 4. STATISTICAL TEST: ANOVA
# Testing if Spending differs by District
print(">>> RUNNING ANOVA TEST")
anova_res <- aov(FOOD_OUTSIDE ~ District_Label, data = df)
print(summary(anova_res))

# 5. STATISTICAL TEST: LINEAR REGRESSION
# Testing Income Elasticity
print(">>> RUNNING LINEAR MODEL SUMMARY")
lm_model <- lm(COFFEE ~ TOINC, data = df)
print(summary(lm_model))

# 6. ADVANCED VISUALIZATION (GGPLOT2)
# Replacing the ugly 'boxplot' with a professional chart
print(">>> GENERATING VALIDATION PLOT")

p <- ggplot(df, aes(x = District_Label, y = FOOD_OUTSIDE, fill = District_Label)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red", outlier.shape = 1) +
    theme_minimal() +
    scale_fill_brewer(palette = "Set2") +
    labs(
        title = "Validation: Spending Distribution by District",
        subtitle = "Source: Synthetic Prototype Data (N=5000)",
        x = "District",
        y = "Monthly Expenditure (PHP)"
    ) +
    theme(legend.position = "none")

# Save as high-res PNG
ggsave("R_Validation_Plot.png", plot = p, width = 8, height = 6)
print(">>> SUCCESS: Validation Plot Saved as 'R_Validation_Plot.png'")
