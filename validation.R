# R Script for Statistical Validation

filename <- "Synthetic_FIES_NCR.csv"

if (file.exists(filename)) {
    print(paste("Loading:", filename))
    df <- read.csv(filename)

    # 1. ANOVA
    print(">>> RUNNING ANOVA: Spending ~ District")
    summary(aov(FOOD_OUTSIDE ~ as.factor(W_PROV), data = df))

    # 2. Linear Model
    print(">>> RUNNING LINEAR MODEL")
    lm_model <- lm(COFFEE ~ TOINC + FSIZE, data = df)
    print(summary(lm_model))

    # 3. Save Validation Plot
    png("R_Validation_Plot.png")
    boxplot(FOOD_OUTSIDE ~ W_PROV,
        data = df,
        main = "R Validation: Spending Distribution by District",
        col = "lightblue"
    )
    dev.off()
    print("Plot Saved.")
} else {
    print("Error: File not found.")
}
