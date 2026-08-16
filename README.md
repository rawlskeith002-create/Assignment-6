[Assignment6 (3) (1).xlsx](https://github.com/user-attachments/files/31123369/Assignment6.3.1.xlsx)

For the last group assignment of BAN statistics we applied a multiple linear regression analysis to our data and we able to identify key variables including depth as well as removing less-impactful points like faint or colored stones, 
before finalizing an output with a supported R^2 and ADJ R^2 values!
This allowed us to determine that for increase in mm of "Z" depth, the standard price increase is $2976.75

Kaggle Dataset source: https://www.kaggle.com/datasets/suryadevsingh/diamond2022

Assignment Overview
In this assignment, your team will apply multiple linear regression analysis to your dataset. You’ll build and refine models to identify which variables meaningfully explain your chosen outcome variable. This assignment will help you develop critical modeling skills including:

Variable selection and elimination based on statistical significance

Model comparison and refinement

Identifying and commenting on multicollinearity

All analysis should be completed in Excel and submitted in a clearly organized workbook.

Requirements
1. Model Setup
Create a tab titled “Model 1 – All Variables.”

Choose one continuous variable as your dependent (outcome) variable.

Include all potential independent variables (categorical and continuous). Note Excel's built-in regression analysis has a limit of 16 independent variables.

Run a full regression model using Excel’s Data Analysis Toolpak.

Include the:

Regression output table

Interpretation of R², adjusted R², and significant vs. insignificant predictors

2. Model Refinement Process
Create additional tabs labeled “Model 2,” “Model 3,” etc. as needed.

If all predictors in Model 1 are significant, you may stop and use that as your final model.

Otherwise, run additional models (at least 5) to refine your model by:

Removing insignificant predictors one at a time

Start with the variable with the highest p-value

Document each step: what variable was removed, and why

3. Final Model
Create a tab titled “Final Model.”

Include:

Full regression output

Interpretation of R² and all model coefficients

Please initially write the interpretation with your own words. Provide this interpretation in your assignment submission. Once you have completed your interpretation use a GenAI toolLinks to an external site. as your assistant to help improve your explanation. You should use it responsibly, ethically and are accountable for its final output. You will fully cite and demonstrate how you used it in your submission. More explicit guidance are provided in the instructions hereDownload here. Discuss in a sentence or two what you gained from using a GenAI tool when explaining the usefulness of your final model. Additionally, if you notice any limitations or challenges in using a GenAI tool, please be sure to include those as well.
Clear explanation of why this is your final model

Confirm that all remaining variables are statistically significant (p < 0.05)

4. Multicollinearity Check
Create a tab titled “Multicollinearity.”

Calculate correlation matrix among continuous independent variables

Comment on:

Any strong correlations (e.g., above 0.7 or 0.8) 

What steps you took to reduce multicollinearity (e.g., removing one of two correlated variables) or why you left them in

5. Commentary Tab
Create a tab titled “Regression Commentary.” In 1–2 paragraphs, reflect on:

How the final model performs

Which predictors were most useful or surprising

Interpret the relationship be the dependent variable and the significant predictors in your final model
How the findings might support a business decision

Limitations or cautions based on your data or model

Tips for Success
Use Excel’s Data Analysis Toolpak to generate regression outputs (Data > Data Analysis > Regression)

Use Excel’s Data Analysis Toolpak to generate a correlation matrix to check correlations

Round coefficients and p-values to 3 decimals

Use comments in cells to explain variable removals or decisions

Clearly highlight your final chosen model in both the tab name and commentary

Submission Format: Upload Excel File
