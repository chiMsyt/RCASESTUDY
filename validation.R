# Run inside RCASESTUDY root
data <- read.csv("Synthetic_FIES_NCR.csv")

# Simple ANOVA validation
print("Running ANOVA on Synthetic Data...")
summary(aov(FOOD_OUTSIDE ~ as.factor(W_PROV), data=data))

# Save Plot
png("R_Validation.png")
boxplot(FOOD_OUTSIDE ~ W_PROV, data=data, main="R Validation: Spending by District")
dev.off()