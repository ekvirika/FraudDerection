# IEEE-CIS Fraud Detection პროექტი

## კონკურსის მოკლე მიმოხილვა
Kaggle-ზე Fraud Detection პრობლემა, რომლის ამოცანაა fraud ტრანზაქციების იდენტიფიცირება.

## ჩვენი მიდგომა
მონაცემთა გაწმენდა, მახასიათებლების ინჟინერია, მოდელის შერჩევა და შედეგების შეფასება MLflow-ის მეშვეობით.

## რეპოზიტორიის სტრუქტურა
- model_experiment_*.ipynb - თითოეული მოდელის ექსპერიმენტები
- model_inference.ipynb - საბოლოო პროგნოზირება
- pipeline/ - შენახული საუკეთესო pipeline-ები
- README.md - ეს ფაილი

## Feature Engineering
- Device/Browser ტიპების დაჯგუფება
- Email Domain-ების გაწმენდა
- TransactionAmt → Log
- Timestamp Features

## Nan დამუშავება


## Feature Selection
- Correlation
- Feature Importance (XGBoost)
- Recursive Feature Elimination

## Model Training
- ტესტირებული მოდელები: LogisticRegression, RandomForest, XGBoost, LightGBM
- Hyperparameter tuning: GridSearchCV

## MLflow Tracking
- ყველა მოდელისთვის ცალკე ექსპერიმენტი
- თითოეული ეტაპი დალოგილია როგორც run
- საუკეთესო მოდელი Model Registry-შია შენახული

## შედეგები


## [MLflow ბმული]()
