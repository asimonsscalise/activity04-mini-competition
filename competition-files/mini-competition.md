Linear Regression Mini-competition
================

``` r
#removing significant missing values for Headline, Source and Facebook/GooglePlus/LinkedIn
  #filter(Title==""|is.na(Title))#0
  #filter(Headline==""|is.na(Headline)) #15
  #filter(Topic==""|is.na(Topic)) #0
  #filter(Source==""|is.na(Source)) #273
  #filter(Facebook<0|GooglePlus<0,LinkedIn<0) #5000

#clean up data
df<-data|>
  #pull date from publishing timestamp
  mutate(date=as.Date(sub(" .*", "", PublishDate),"%m/%d/%Y"),
         #create a weekday variable in case day of the week article was published contributes to sentiment
         DoW=wday(date, abbr = FALSE),
         #track word counts in title and headline in case they contribute to sentiment
         titleWC=str_count(Title, '\\w+'),
         headlineWC=str_count(Headline, '\\w+'))|>
  #filter to remove missing values found above
  filter(Facebook>=0,
         GooglePlus>=0,
         LinkedIn>=0,
         Headline!="",
         !is.na(Headline),
         Source!="",
         Source!=" ")
#There are so many unique sources, we want to get a count of publishing frequency to see if this allows us to remove sources from the model
counts<-df|>
  count(Source)

#join counts back to original df
df<-left_join(df,counts,by="Source")

#sort the counts for source into bins based on publishing frequency
df<-df%>%
  mutate(
    sourcegroup=as.factor(case_when(
    n<=302~"low_publish_rate",
    n>=303&n<=603~"med_low_publish_rate",
    n>=604&n<=904~"med_publish_rate",
    n>=905&n<=1205~"med_high_publish_rate",
    n>=1206~"high_publish_rate")),
    #extract month/year from date
    month=as.factor(format(as.Date(date), "%Y-%m")),
    #set other variables to factors
    Topic=as.factor(Topic),
    DoW=as.factor(DoW))%>%
  select(-c(IDLink,Title,Headline,PublishDate,Source,date,n))


#see a summary of the df data
summary(df)
```

    ##        Topic       SentimentTitle      SentimentHeadline     Facebook      
    ##  economy  :29289   Min.   :-0.950694   Min.   :-0.75543   Min.   :    0.0  
    ##  microsoft:18112   1st Qu.:-0.079057   1st Qu.:-0.11396   1st Qu.:    1.0  
    ##  obama    :26598   Median : 0.000000   Median :-0.02606   Median :    8.0  
    ##  palestine: 7280   Mean   :-0.005413   Mean   :-0.02753   Mean   :  129.9  
    ##                    3rd Qu.: 0.065693   3rd Qu.: 0.05964   3rd Qu.:   44.0  
    ##                    Max.   : 0.962354   Max.   : 0.96465   Max.   :49211.0  
    ##                                                                            
    ##    GooglePlus          LinkedIn       DoW          titleWC      
    ##  Min.   :   0.000   Min.   :   0.00   1: 7172   Min.   : 1.000  
    ##  1st Qu.:   0.000   1st Qu.:   0.00   2:14379   1st Qu.: 8.000  
    ##  Median :   0.000   Median :   0.00   3:14637   Median : 9.000  
    ##  Mean   :   4.333   Mean   :  15.56   4:13648   Mean   : 9.264  
    ##  3rd Qu.:   2.000   3rd Qu.:   4.00   5:13343   3rd Qu.:11.000  
    ##  Max.   :1267.000   Max.   :6362.00   6:12059   Max.   :25.000  
    ##                                       7: 6041                   
    ##    headlineWC                   sourcegroup        month      
    ##  Min.   : 0.00   high_publish_rate    : 1506   2016-01:12283  
    ##  1st Qu.:22.00   low_publish_rate     :57887   2016-03:11506  
    ##  Median :24.00   med_high_publish_rate: 2182   2015-12:11403  
    ##  Mean   :27.16   med_low_publish_rate :12664   2016-02:10718  
    ##  3rd Qu.:26.00   med_publish_rate     : 7040   2016-04:10601  
    ##  Max.   :87.00                                 2016-05:10285  
    ##                                                (Other):14483

``` r
#look at a correlation plot of all variables (one-hot encoded to allow to correlation between categorical variables)
#model.matrix(~0+., data=df)%>% 
#  cor(use="pairwise.complete.obs") %>% 
#  ggcorrplot(show.diag=FALSE, method="square",type="upper", lab=TRUE, lab_size=2)
```

``` r
set.seed(456)

#remove sentiment headline from the model
sentimentTitle<-df%>%
  select(-SentimentHeadline)

#build a recipe using the all variables as predictors
titlerecipe <- recipe(
  #set your formula 
  SentimentTitle ~ .,
  data = sentimentTitle) 
#create a flow that one-hot encodes
titleflow<-titlerecipe %>%
  step_dummy(all_nominal_predictors(),one_hot=TRUE) %>% #one-hot encode
  prep()
#bake the recipe into dataset 
test<-bake(titleflow,new_data = NULL)


#using rsample package, split data with 25% to test and 75% to training
split_Title <- initial_split(test,prop = 3/4)
#set your training data based on split parameters
train_title <- training(split_Title)
#set your testing data based on split parameters
test_title <- testing(split_Title)


#set model, set engine
linear_mod_title <- linear_reg() %>%
      set_engine("lm")

#fit model with prepped data from recipe
linear_title <- fit(formula = SentimentTitle ~., linear_mod_title, data = train_title)
#predict on test data already prepped from recipe
lm_predicted_title <- predict(linear_title, new_data = test_title)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response", : prediction from rank-deficient fit; consider predict(.,
    ## rankdeficient="NA")

``` r
#get predictions and actual values
lm_title_evaluation <- as.data.frame(cbind(lm_predicted_title, test_title$SentimentTitle))
colnames(lm_title_evaluation) <- c("Predicted", "Actual")

#evaluation metrics
rmse(lm_title_evaluation, truth = Actual, estimate = Predicted)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.137

``` r
rsq(lm_title_evaluation, truth = Actual, estimate = Predicted)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rsq     standard     0.00465

``` r
mae(lm_title_evaluation, truth = Actual, estimate = Predicted)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 mae     standard      0.0975

``` r
tidy(linear_title)
```

    ## # A tibble: 31 × 5
    ##    term                 estimate    std.error statistic   p.value
    ##    <chr>                   <dbl>        <dbl>     <dbl>     <dbl>
    ##  1 (Intercept)      0.0128        0.00752        1.70    8.97e- 2
    ##  2 Facebook         0.0000000658  0.000000961    0.0685  9.45e- 1
    ##  3 GooglePlus      -0.000119      0.0000343     -3.46    5.49e- 4
    ##  4 LinkedIn        -0.00000660    0.00000739    -0.893   3.72e- 1
    ##  5 titleWC         -0.00249       0.000249     -10.0     1.39e-23
    ##  6 headlineWC      -0.0000631     0.0000480     -1.31    1.89e- 1
    ##  7 Topic_economy    0.00892       0.00209        4.26    2.03e- 5
    ##  8 Topic_microsoft  0.0222        0.00223        9.97    2.06e-23
    ##  9 Topic_obama      0.0187        0.00213        8.78    1.75e-18
    ## 10 Topic_palestine NA            NA             NA      NA       
    ## # ℹ 21 more rows

``` r
ggplot(lm_title_evaluation, aes(x = Predicted, y = Actual, color=Predicted)) + 
                 geom_point(alpha=0.3) + 
                 geom_abline(lty = 2, color = "gray80", size = 1.5)+
                 labs(color=NULL,
                      title = "Predicted and True Title Sentiment")
```

    ## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
    ## ℹ Please use `linewidth` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

![](mini-competition_files/figure-gfm/first%20title%20model-1.png)<!-- -->

``` r
test<-test%>%
  select(GooglePlus,titleWC,SentimentTitle,Topic_economy,Topic_microsoft,Topic_obama,Topic_palestine,DoW_X4,month_X2015.11)

split_Title <- initial_split(test,prop = 3/4)
#set your training data based on split parameters
train_title <- training(split_Title)
#set your testing data based on split parameters
test_title <- testing(split_Title)


#set model, set engine
linear_mod_title <- linear_reg() %>%
      set_engine("lm")

#fit model with prepped data from recipe
linear_title <- fit(formula = SentimentTitle ~., linear_mod_title, data = train_title)
#predict on test data already prepped from recipe
lm_predicted_title <- predict(linear_title, new_data = test_title)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response", : prediction from rank-deficient fit; consider predict(.,
    ## rankdeficient="NA")

``` r
#get predictions and actual values
lm_title_evaluation <- as.data.frame(cbind(lm_predicted_title, test_title$SentimentTitle))
colnames(lm_title_evaluation) <- c("Prediction", "Actual")

#evaluation metrics
rmse(lm_title_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.136

``` r
rsq(lm_title_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rsq     standard     0.00393

``` r
mae(lm_title_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 mae     standard      0.0970

``` r
tidy(linear_title)
```

    ## # A tibble: 9 × 5
    ##   term             estimate  std.error statistic   p.value
    ##   <chr>               <dbl>      <dbl>     <dbl>     <dbl>
    ## 1 (Intercept)      0.00336   0.00283        1.19  2.35e- 1
    ## 2 GooglePlus      -0.000104  0.0000275     -3.79  1.52e- 4
    ## 3 titleWC         -0.00256   0.000247     -10.3   4.53e-25
    ## 4 Topic_economy    0.00878   0.00207        4.24  2.27e- 5
    ## 5 Topic_microsoft  0.0237    0.00221       10.7   6.42e-27
    ## 6 Topic_obama      0.0189    0.00210        9.02  2.01e-19
    ## 7 Topic_palestine NA        NA             NA    NA       
    ## 8 DoW_X4           0.00462   0.00148        3.13  1.77e- 3
    ## 9 month_X2015.11  -0.00439   0.00195       -2.25  2.43e- 2

``` r
ggplot(lm_title_evaluation, aes(x = Prediction, y = Actual, color=Prediction)) + 
                 geom_point(alpha=0.3) + 
                 geom_abline(lty = 2, color = "gray80", size = 1.5)+
                 labs(color=NULL,
                      title = "Predicted and True Title Sentiment")
```

![](mini-competition_files/figure-gfm/title%20model%202-1.png)<!-- -->

``` r
sentimentHL<-df%>%
  select(-SentimentTitle)
#build a recipe 
HLrecipe <- recipe(
  #set your formula 
  SentimentHeadline ~ .,
  data = sentimentHL) 
#create a flow that one-hot encodes
HLflow<-HLrecipe %>%
  step_dummy(all_nominal_predictors(),one_hot=TRUE) %>% #one-hot encode
  prep()
#bake the recipe into dataset 
test<-bake(HLflow,new_data = NULL)


#using rsample package, split data with 25% to test and 75% to training
split_HL <- initial_split(test,prop = 3/4)
#set your training data based on split parameters
train_HL<- training(split_HL)
#set your testing data based on split parameters
test_HL<- testing(split_HL)


#set model, set engine
linear_mod_HL <- linear_reg() %>%
      set_engine("lm")

#fit model with prepped data from recipe
linear_HL<- fit(formula = SentimentHeadline ~., linear_mod_HL, data = train_HL)
#predict on test data already prepped from recipe
lm_predicted_HL <- predict(linear_HL, new_data = test_HL)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response", : prediction from rank-deficient fit; consider predict(.,
    ## rankdeficient="NA")

``` r
#get predictions and actual values
lm_HL_evaluation <- as.data.frame(cbind(lm_predicted_HL, test_HL$SentimentHeadline))
colnames(lm_HL_evaluation) <- c("Prediction", "Actual")

#evaluation metrics
rmse(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.141

``` r
rsq(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rsq     standard      0.0122

``` r
mae(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 mae     standard       0.108

``` r
tidy(linear_HL)
```

    ## # A tibble: 31 × 5
    ##    term               estimate    std.error statistic   p.value
    ##    <chr>                 <dbl>        <dbl>     <dbl>     <dbl>
    ##  1 (Intercept)     -0.00815     0.00769        -1.06   2.89e- 1
    ##  2 Facebook        -0.00000173  0.000000975    -1.78   7.57e- 2
    ##  3 GooglePlus      -0.0000550   0.0000344      -1.60   1.10e- 1
    ##  4 LinkedIn        -0.00000963  0.00000772     -1.25   2.12e- 1
    ##  5 titleWC         -0.000234    0.000258       -0.905  3.65e- 1
    ##  6 headlineWC      -0.000746    0.0000495     -15.1    3.43e-51
    ##  7 Topic_economy    0.00695     0.00218         3.19   1.41e- 3
    ##  8 Topic_microsoft  0.0343      0.00231        14.8    9.69e-50
    ##  9 Topic_obama      0.0295      0.00221        13.4    9.88e-41
    ## 10 Topic_palestine NA          NA              NA     NA       
    ## # ℹ 21 more rows

``` r
ggplot(lm_HL_evaluation, aes(x = Prediction, y = Actual, color=Prediction)) + 
                 geom_point(alpha=0.3) + 
                 geom_abline(lty = 2, color = "gray80", size = 1.5)+
                 labs(color=NULL,
                      title = "Predicted and True Title Sentiment")
```

![](mini-competition_files/figure-gfm/first%20headline%20model-1.png)<!-- -->

``` r
test_HL<-test_HL%>%
  select(titleWC,SentimentHeadline,Topic_economy,Topic_microsoft,Topic_obama,Topic_palestine,month_X2015.11,month_X2015.12,month_X2016.01,month_X2016.02,month_X2016.03,month_X2016.04,month_X2016.05,month_X2016.06,month_X2016.07)

split_HL <- initial_split(test_HL,prop = 3/4)
#set your training data based on split parameters
train_HL <- training(split_HL)
#set your testing data based on split parameters
test_HL <- testing(split_HL)


#set model, set engine
linear_mod_HL <- linear_reg() %>%
      set_engine("lm")

#fit model with prepped data from recipe
linear_HL <- fit(formula = SentimentHeadline ~., linear_mod_HL, data = train_HL)
#predict on test data already prepped from recipe
lm_predicted_HL <- predict(linear_HL, new_data = test_HL)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response", : prediction from rank-deficient fit; consider predict(.,
    ## rankdeficient="NA")

``` r
#get predictions and actual values
lm_HL_evaluation <- as.data.frame(cbind(lm_predicted_HL, test_HL$SentimentHeadline))
colnames(lm_HL_evaluation) <- c("Prediction", "Actual")

#evaluation metrics
rmse(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.140

``` r
rsq(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rsq     standard     0.00920

``` r
mae(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 mae     standard       0.109

``` r
tidy(linear_HL)
```

    ## # A tibble: 15 × 5
    ##    term             estimate std.error statistic   p.value
    ##    <chr>               <dbl>     <dbl>     <dbl>     <dbl>
    ##  1 (Intercept)     -0.0274    0.0142     -1.93    5.31e- 2
    ##  2 titleWC         -0.000834  0.000511   -1.63    1.02e- 1
    ##  3 Topic_economy    0.00527   0.00430     1.23    2.21e- 1
    ##  4 Topic_microsoft  0.0279    0.00458     6.09    1.13e- 9
    ##  5 Topic_obama      0.0273    0.00434     6.29    3.21e-10
    ##  6 Topic_palestine NA        NA          NA      NA       
    ##  7 month_X2015.11  -0.0136    0.0136     -1.00    3.16e- 1
    ##  8 month_X2015.12  -0.00894   0.0134     -0.669   5.04e- 1
    ##  9 month_X2016.01  -0.00736   0.0133     -0.552   5.81e- 1
    ## 10 month_X2016.02  -0.00745   0.0134     -0.556   5.79e- 1
    ## 11 month_X2016.03  -0.0102    0.0134     -0.762   4.46e- 1
    ## 12 month_X2016.04  -0.00967   0.0134     -0.722   4.70e- 1
    ## 13 month_X2016.05  -0.0137    0.0134     -1.02    3.06e- 1
    ## 14 month_X2016.06  -0.00118   0.0136     -0.0869  9.31e- 1
    ## 15 month_X2016.07  NA        NA          NA      NA

``` r
ggplot(lm_HL_evaluation, aes(x = Prediction, y = Actual, color=Prediction)) + 
                 geom_point(alpha=0.3) + 
                 geom_abline(lty = 2, color = "gray80", size = 1.5)+
                 labs(color=NULL,
                      title = "Predicted and True HL Sentiment")
```

![](mini-competition_files/figure-gfm/HEadline%20model%202-1.png)<!-- -->

``` r
#build a recipe using the all variables as predictors
titlerecipe <- recipe(
  #set your formula 
  SentimentTitle ~ .,
  data = df) 
#create a flow that one-hot encodes
titleflow<-titlerecipe %>%
  step_dummy(all_nominal_predictors(),one_hot=TRUE) %>% #one-hot encode
  prep()
#bake the recipe into dataset 
test<-bake(titleflow,new_data = NULL)


#using rsample package, split data with 25% to test and 75% to training
split_Title <- initial_split(test,prop = 3/4)
#set your training data based on split parameters
train_title <- training(split_Title)
#set your testing data based on split parameters
test_title <- testing(split_Title)


#set model, set engine
linear_mod_title <- linear_reg() %>%
      set_engine("lm")

#fit model with prepped data from recipe
linear_title <- fit(formula = SentimentTitle ~., linear_mod_title, data = train_title)
#predict on test data already prepped from recipe
lm_predicted_title <- predict(linear_title, new_data = test_title)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response", : prediction from rank-deficient fit; consider predict(.,
    ## rankdeficient="NA")

``` r
#get predictions and actual values
lm_title_evaluation <- as.data.frame(cbind(lm_predicted_title, test_title$SentimentTitle))
colnames(lm_title_evaluation) <- c("Predicted", "Actual")

#evaluation metrics
rmse(lm_title_evaluation, truth = Actual, estimate = Predicted)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.134

``` r
rsq(lm_title_evaluation, truth = Actual, estimate = Predicted)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rsq     standard      0.0365

``` r
mae(lm_title_evaluation, truth = Actual, estimate = Predicted)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 mae     standard      0.0972

``` r
tidy(linear_title)
```

    ## # A tibble: 32 × 5
    ##    term                  estimate  std.error statistic  p.value
    ##    <chr>                    <dbl>      <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)        0.00914     0.00732       1.25   2.12e- 1
    ##  2 SentimentHeadline  0.171       0.00385      44.5    0       
    ##  3 Facebook          -0.000000498 0.00000112   -0.445  6.56e- 1
    ##  4 GooglePlus        -0.0000606   0.0000341    -1.78   7.57e- 2
    ##  5 LinkedIn          -0.00000444  0.00000681   -0.652  5.14e- 1
    ##  6 titleWC           -0.00245     0.000245     -9.99   1.77e-23
    ##  7 headlineWC         0.00000446  0.0000473     0.0942 9.25e- 1
    ##  8 Topic_economy      0.00882     0.00206       4.28   1.88e- 5
    ##  9 Topic_microsoft    0.0188      0.00220       8.55   1.30e-17
    ## 10 Topic_obama        0.0156      0.00210       7.42   1.15e-13
    ## # ℹ 22 more rows

``` r
ggplot(lm_title_evaluation, aes(x = Predicted, y = Actual, color=Predicted)) + 
                 geom_point(alpha=0.3) + 
                 geom_abline(lty = 2, color = "gray80", size = 1.5)+
                 labs(color=NULL,
                      title = "Predicted and True Title Sentiment")
```

![](mini-competition_files/figure-gfm/title%20with%20headline%20sentiment-1.png)<!-- -->

``` r
#build a recipe 
HLrecipe <- recipe(
  #set your formula 
  SentimentHeadline ~ .,
  data = df) 
#create a flow that one-hot encodes
HLflow<-HLrecipe %>%
  step_dummy(all_nominal_predictors(),one_hot=TRUE) %>% #one-hot encode
  prep()
#bake the recipe into dataset 
test<-bake(HLflow,new_data = NULL)


#using rsample package, split data with 25% to test and 75% to training
split_HL <- initial_split(test,prop = 3/4)
#set your training data based on split parameters
train_HL<- training(split_HL)
#set your testing data based on split parameters
test_HL<- testing(split_HL)


#set model, set engine
linear_mod_HL <- linear_reg() %>%
      set_engine("lm")

#fit model with prepped data from recipe
linear_HL<- fit(formula = SentimentHeadline ~., linear_mod_HL, data = train_HL)
#predict on test data already prepped from recipe
lm_predicted_HL <- predict(linear_HL, new_data = test_HL)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response", : prediction from rank-deficient fit; consider predict(.,
    ## rankdeficient="NA")

``` r
#get predictions and actual values
lm_HL_evaluation <- as.data.frame(cbind(lm_predicted_HL, test_HL$SentimentHeadline))
colnames(lm_HL_evaluation) <- c("Prediction", "Actual")

#evaluation metrics
rmse(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rmse    standard       0.139

``` r
rsq(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 rsq     standard      0.0489

``` r
mae(lm_HL_evaluation, truth = Actual, estimate = Prediction)
```

    ## # A tibble: 1 × 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 mae     standard       0.107

``` r
tidy(linear_HL)
```

    ## # A tibble: 32 × 5
    ##    term                estimate  std.error statistic  p.value
    ##    <chr>                  <dbl>      <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)     -0.00102     0.00748       -0.137 8.91e- 1
    ##  2 SentimentTitle   0.180       0.00412       43.6   0       
    ##  3 Facebook        -0.000000711 0.00000100    -0.709 4.78e- 1
    ##  4 GooglePlus      -0.0000290   0.0000344     -0.844 3.99e- 1
    ##  5 LinkedIn        -0.00000625  0.00000688    -0.909 3.64e- 1
    ##  6 titleWC          0.0000712   0.000254       0.281 7.79e- 1
    ##  7 headlineWC      -0.000809    0.0000487    -16.6   1.00e-61
    ##  8 Topic_economy    0.00454     0.00213        2.13  3.33e- 2
    ##  9 Topic_microsoft  0.0276      0.00227       12.2   5.96e-34
    ## 10 Topic_obama      0.0250      0.00217       11.6   7.41e-31
    ## # ℹ 22 more rows

``` r
ggplot(lm_HL_evaluation, aes(x = Prediction, y = Actual, color=Prediction)) + 
                 geom_point(alpha=0.3) + 
                 geom_abline(lty = 2, color = "gray80", size = 1.5)+
                 labs(color=NULL,
                      title = "Predicted and True Title Sentiment")
```

![](mini-competition_files/figure-gfm/headline%20with%20title%20sentiment-1.png)<!-- -->
