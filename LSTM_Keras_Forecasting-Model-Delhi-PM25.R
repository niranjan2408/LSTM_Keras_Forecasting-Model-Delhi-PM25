library(tensorflow)
library(keras)
library(readxl)
library(ggplot2)
library(padr)
library(dplyr)
library(imputeTS)
library(timetk)
library(tibbletime)
library(tibble)
library(rlang)
library(tidyquant)
library(rsample)
library(forcats)
library(glue)
library(tidyverse)
library(cowplot)
library(recipes)
library(yardstick)


delhi<-read_excel("D:/Data Science Project/DelhiPM25/Delhi (1).xlsx",sheet = "Delhi",col_types = c("date","numeric"),skip = 2)
delhi <- delhi[rev(1:nrow(delhi)),] #reversing the data entries
date_time <-pad(as.data.frame(delhi$date))
colnames(date_time) <- 'date'
updated_data<- full_join(date_time,delhi)
updated_data$na_seasplit_ma<- na_seasplit(updated_data$pm25,algorithm = "ma",find_frequency=TRUE )
model_data <- updated_data[,c(1,3)]
colnames(model_data) <- c("Date","PM25")


model_data <- model_data %>% tk_tbl()

tidy_acf <- function(data, PM25, lags = 0:48) {
  value_expr <- enquo(PM25)
  acf_values <- data %>% pull(PM25) %>% acf(lag.max = tail(lags, 1), plot = FALSE) %>% .$acf %>% .[,,1]
  ret <- tibble(acf = acf_values) %>% rowid_to_column(var = "lag") %>% mutate(lag = lag - 1) %>% filter(lag %in% lags)
  return(ret)
}

max_lag <- 48
model_data %>%  tidy_acf(PM25, lags = 0:max_lag)

model_data %>%
  tidy_acf(PM25, lags = 0:max_lag) %>%
  ggplot(aes(lag, acf)) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
  geom_vline(xintercept = 24, size = 2, color = palette_light()[[2]]) +
  annotate("text", label = "1 Day Mark", x = 22, y = 0.9, 
           color = palette_light()[[2]], size = 6, hjust = 0) +
  theme_tq() +
  labs(title = "ACF: PM25")


model_data %>%
  tidy_acf(PM25, lags = 1:24) %>%
  ggplot(aes(lag, acf)) +
  geom_vline(xintercept = 24, size = 2, color = palette_light()[[2]]) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
  geom_point(color = palette_light()[[1]], size = 2) +
  geom_label(aes(label = acf %>% round(2)), vjust = -1,
             color = palette_light()[[1]]) +
  annotate("text", label = "1 Day Mark", x = 22.5, y = 0.9, 
           color = palette_light()[[2]], size = 5, hjust = 0) +
  theme_tq() +
  labs(title = "ACF: Sunspots",
       subtitle = "Zoomed in on Lags 1 to 24")

optimal_lag_setting <- model_data %>%  tidy_acf(PM25, lags = 1:24) %>%  filter(acf == max(acf)) %>% pull(lag)
optimal_lag_setting


periods_train <- 24 * 28
periods_test  <- 24 * 7
skip_span     <- 24 * 7

rolling_origin_resamples <- rolling_origin(model_data,initial= periods_train, assess= periods_test,  cumulative = FALSE, skip= skip_span)
rolling_origin_resamples

plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
  
  # Manipulate data
  train_tbl <- training(split) %>% add_column(key = "training") 
  test_tbl  <- testing(split) %>% add_column(key = "testing") 
  
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>% as_tbl_time(index = Date) %>%  mutate(key = fct_relevel(key,"training","testing"))
  
  # Collect attributes
  train_time_summary <- train_tbl %>% tk_index() %>%  tk_get_timeseries_summary()
  test_time_summary <- test_tbl %>%   tk_index() %>%  tk_get_timeseries_summary()
  
  # Visualize
  g <- data_manipulated %>%
    ggplot(aes(x = Date, y = PM25,color = key)) +
    geom_line(size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs( title    = glue("Split: {split$id}"),subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
      y = "", x = ""    ) +    theme(legend.position = "none") 
  
  if (expand_y_axis) {
    
    model_data_summary <- model_data %>% tk_index() %>%tk_get_timeseries_summary()
    g <- g + scale_x_date(limits = c(model_data_summary$start, model_data_summary$end))
  }
  
  return(g)
}


rolling_origin_resamples$splits[[1]] %>%  plot_split(expand_y_axis = F) +  theme(legend.position = "bottom")


# Plotting function that scales to all splits 
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, ncol = 3, alpha = 1, size = 1, base_size = 14,title = "Sampling Plan") {
  
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%mutate(gg_plots = map(splits, plot_split,expand_y_axis = expand_y_axis,alpha = alpha, base_size = base_size))
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  p_title <- ggdraw() + draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  return(g)
  
}


rolling_origin_resamples %>%plot_sampling_plan(expand_y_axis = F, ncol = 3, alpha = 1, size = 1, base_size = 10,title = "Backtesting Strategy: Zoomed In")

#Single LSTM
#For the single LSTM model, we'll select and visualize the split for the most recent time sample/slice (Slice10). 
#The 11th split contains the most recent data.

split    <- rolling_origin_resamples$splits[[11]]
split_id <- rolling_origin_resamples$id[[11]]

plot_split(split, expand_y_axis = FALSE, size = 0.5) +  theme(legend.position = "bottom") +  ggtitle(glue("Split: {split_id}"))


#Data Setup
df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing")
) %>% 
  as_tbl_time(index = Date)

df

#Preprocessing With Recipes
#The LSTM algorithm requires the input data to be centered and scaled. 
#combination of step_sqrt to transform the data and reduce the presence of outliers and step_center and step_scale to center and scale the data. 
#The data is processed/transformed using the bake() function

rec_obj <- recipe(PM25 ~ ., df) %>%  step_sqrt(PM25) %>%  step_center(PM25) %>%  step_scale(PM25) %>%  prep()
df_processed_tbl <- bake(rec_obj, df)
df_processed_tbl

#Capturing the center/scale history so we can invert the center and scaling after modeling. 
#The square-root transformation can be inverted by squaring the inverted center/scale values.

center_history <- rec_obj$steps[[2]]$means["PM25"]
scale_history  <- rec_obj$steps[[3]]$sds["PM25"]
c("center" = center_history, "scale" = scale_history)


#LSTM Plan
# Model inputs

lag_setting  <- 1 
batch_size   <- 24
train_length <- 336
tsteps       <- 1
epochs       <- 100

# Training Set
lag_train_tbl <- df_processed_tbl %>%  mutate(PM25_lag = lag(PM25, n = lag_setting)) %>%  filter(!is.na(PM25_lag)) %>%
  filter(key == "training") %>%  tail(train_length)

x_train_vec <- lag_train_tbl$PM25_lag
x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))

y_train_vec <- lag_train_tbl$PM25
y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))


# Testing Set
lag_test_tbl <- df_processed_tbl %>%  mutate(PM25_lag = lag(PM25, n = lag_setting)  ) %>%  filter(!is.na(PM25_lag)) %>%  
  filter(key == "testing")

x_test_vec <- lag_test_tbl$PM25_lag
x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))

y_test_vec <- lag_test_tbl$PM25
y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))



#Building The LSTM Model

model <- keras_model_sequential()
         
model %>%
  layer_lstm(units            = 24, 
             input_shape      = c(tsteps, 1), 
             batch_size       = batch_size,
             return_sequences = TRUE, 
             stateful         = TRUE) %>% 
  layer_lstm(units            = 24, 
             return_sequences = FALSE, 
             stateful         = TRUE) %>% 
  layer_dense(units = 1)

model %>%   compile(loss = 'mae', optimizer = 'adam')
model


for (i in 1:epochs) {
  model %>% fit(x          = x_train_arr, 
                y          = y_train_arr, 
                batch_size = batch_size,
                epochs     = 1, 
                verbose    = 1, 
                shuffle    = FALSE)
  model %>% reset_states()
  cat("Epoch: ", i)
  
}


# Make Predictions
pred_out <- model %>%  predict(x_test_arr, batch_size = batch_size) %>%  .[,1]   

# Retransform values
pred_tbl <- tibble(
  Date   = lag_test_tbl$Date,
  PM25   = (pred_out * scale_history + center_history)^2
)   


# Combine actual data with predictions
tbl_1 <- df_trn %>%  add_column(key = "actual")

tbl_2 <- df_tst %>%  add_column(key = "actual")

tbl_3 <- pred_tbl %>%  add_column(key = "predict")


# Create time_bind_rows() to solve dplyr issue
time_bind_rows <- function(data_1, data_2, index) {
  index_expr <- enquo(index)
  bind_rows(data_1, data_2) %>%
    as_tbl_time(index = !! index_expr)
}

ret <- list(tbl_1, tbl_2, tbl_3) %>%  reduce(time_bind_rows, index = Date) %>%  arrange(key, Date) %>%  mutate(key = as_factor(key))

ret


calc_rmse <- function(prediction_tbl) {
  
  rmse_calculation <- function(data) {
    data %>%  spread(key = key, value = PM25) %>%  select(-Date) %>% filter(!is.na(predict)) %>%
      rename(
        truth    = actual,
        estimate = predict
      ) %>%  rmse(truth, estimate)%>% select(.estimate) %>% as.matrix()
  }
  safe_rmse <- possibly(rmse_calculation, otherwise = NA)
  
  safe_rmse(prediction_tbl)
  
}

calc_rmse(ret)


#Visualizing The Single Prediction

plot_prediction <- function(data, id, alpha = 1, size = 2, base_size = 14) {
  
  rmse_val <- calc_rmse(data)
  
  g <- data %>%
    ggplot(aes(Date, PM25, color = key)) +
    geom_point(alpha = alpha, size = size) + 
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    theme(legend.position = "none") +
    labs(
      title = glue("{id}, RMSE: {round(rmse_val, digits = 1)}"),
      x = "", y = ""
    )
  
  return(g)
}

ret %>%   plot_prediction(id = split_id, alpha = 0.6) +  theme(legend.position = "bottom")



#Creating An LSTM Prediction Function

predict_keras_lstm <- function(split, epochs = 100, ...) {
  
  lstm_prediction <- function(split, epochs, ...) {
    
    # 5.1.2 Data Setup
    df_trn <- training(split)
    df_tst <- testing(split)
    
    df <- bind_rows(
      df_trn %>% add_column(key = "training"),
      df_tst %>% add_column(key = "testing")
    ) %>% 
      as_tbl_time(index = Date)
    
    # 5.1.3 Preprocessing
    rec_obj <- recipe(PM25 ~ ., df) %>%
      step_sqrt(PM25) %>%
      step_center(PM25) %>%
      step_scale(PM25) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    center_history <- rec_obj$steps[[2]]$means["PM25"]
    scale_history  <- rec_obj$steps[[3]]$sds["PM25"]
    
    # 5.1.4 LSTM Plan
    lag_setting  <- 1 # = nrow(df_tst)
    batch_size   <- 24
    train_length <- 336
    tsteps       <- 1
    epochs       <- epochs
    
    # 5.1.5 Train/Test Setup
    lag_train_tbl <- df_processed_tbl %>%
      mutate(PM25_lag = lag(PM25, n = lag_setting)) %>%
      filter(!is.na(PM25_lag)) %>%
      filter(key == "training") %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$PM25_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
    
    y_train_vec <- lag_train_tbl$PM25
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
    
    lag_test_tbl <- df_processed_tbl %>%
      mutate(
        PM25_lag = lag(PM25, n = lag_setting)
      ) %>%
      filter(!is.na(PM25_lag)) %>%
      filter(key == "testing")
    
    x_test_vec <- lag_test_tbl$PM25_lag
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
    
    y_test_vec <- lag_test_tbl$PM25
    y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))
    
    # 5.1.6 LSTM Model
    model <- keras_model_sequential()
    
    model %>%
      layer_lstm(units            = 24, 
                 input_shape      = c(tsteps, 1), 
                 batch_size       = batch_size,
                 return_sequences = TRUE, 
                 stateful         = TRUE) %>% 
      layer_lstm(units            = 24, 
                 return_sequences = FALSE, 
                 stateful         = TRUE) %>% 
      layer_dense(units = 1)
    
    model %>% 
      compile(loss = 'mae', optimizer = 'adam')
    
    # 5.1.7 Fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x          = x_train_arr, 
                    y          = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1, 
                    shuffle    = FALSE)
      
      model %>% reset_states()
      cat("Epoch: ", i)
      
    }
    
    # 5.1.8 Predict and Return Tidy Data
    # Make Predictions
    pred_out <- model %>% predict(x_test_arr, batch_size = batch_size) %>%.[,1] 
    
    # Retransform values
    pred_tbl <- tibble(
      Date   = lag_test_tbl$Date,
      PM25   = (pred_out * scale_history + center_history)^2
    ) 
    
    # Combine actual data with predictions
    tbl_1 <- df_trn %>%
      add_column(key = "actual")
    
    tbl_2 <- df_tst %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Create time_bind_rows() to solve dplyr issue

    
    ret <- bind_rows(tbl_1, tbl_2, tbl_3) %>% as_tbl_time(index = Date) %>%  mutate(key = fct_relevel(key,"actual","predict"))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  safe_lstm(split, epochs, ...)
  
}

predict_keras_lstm(split, epochs = 10)



sample_predictions_lstm_tbl <- rolling_origin_resamples %>%  mutate(predict = map(splits, predict_keras_lstm, epochs = 100))
sample_predictions_lstm_tbl

sample_rmse_tbl <- sample_predictions_lstm_tbl %>%  mutate(rmse = map_dbl(predict, calc_rmse)) %>%  select(id, rmse)
sample_rmse_tbl



plot_predictions <- function(sampling_tbl, predictions_col,ncol = 3, alpha = 1, size = 2, base_size = 14,
                             title = "Backtested Predictions") {
  
  predictions_col_expr <- enquo(predictions_col)
  
  # Map plot_split() to sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map2(!! predictions_col_expr, id, 
                           .f        = plot_prediction, 
                           alpha     = alpha, 
                           size      = size, 
                           base_size = base_size)) 
  
  # Make plots with cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}

sample_predictions_lstm_tbl %>%
  plot_predictions(predictions_col = predict, alpha = 0.5, size = 1, base_size = 10,
                   title = "Keras Stateful LSTM: Backtested Predictions")





predict_keras_lstm_future <- function(data, epochs = 100, ...) {
  
  lstm_prediction <- function(data, epochs, ...) {
    
    # 5.1.2 Data Setup (MODIFIED)
    df <- model_data
    
    # 5.1.3 Preprocessing
    rec_obj <- recipe(PM25 ~ ., df) %>%
      step_sqrt(PM25) %>%
      step_center(PM25) %>%
      step_scale(PM25) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    center_history <- rec_obj$steps[[2]]$means["PM25"]
    scale_history  <- rec_obj$steps[[3]]$sds["PM25"]
    
    # 5.1.4 LSTM Plan
    lag_setting  <- 1 
    batch_size   <- 24
    train_length <- 336
    tsteps       <- 1
    epochs       <- epochs
    
    # 5.1.5 Train Setup (MODIFIED)
    lag_train_tbl <- df_processed_tbl %>%
      mutate(PM25_lag = lag(PM25, n = lag_setting)) %>%
      filter(!is.na(PM25_lag)) %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$PM25_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
    
    y_train_vec <- lag_train_tbl$PM25
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
    
    x_test_vec <- y_train_vec 
    
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
    
    # 5.1.6 LSTM Model
    model <- keras_model_sequential()
    
    model %>%
      layer_lstm(units            = 7, 
                 input_shape      = c(tsteps, 1), 
                 batch_size       = batch_size,
                 return_sequences = TRUE, 
                 stateful         = TRUE) %>% 
      layer_lstm(units            = 7, 
                 return_sequences = FALSE, 
                 stateful         = TRUE) %>% 
      layer_dense(units = 1)
    
    model %>% 
      compile(loss = 'mae', optimizer = 'adam')
    
    # 5.1.7 Fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x          = x_train_arr, 
                    y          = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1, 
                    shuffle    = FALSE)
      
      model %>% reset_states()
      cat("Epoch: ", i)
      
    }
    
    # 5.1.8 Predict and Return Tidy Data (MODIFIED)
    # Make Predictions
    pred_out <- model %>% predict(x_test_arr, batch_size = batch_size) %>%.[,1] 
    
    # Make future index using tk_make_future_timeseries()
    idx <- data %>%
      tk_index() %>%
      tk_make_future_timeseries(n_future = 336)
    
    # Retransform values
    pred_tbl <- tibble(
      Date   = idx,
      PM25   = (pred_out * scale_history + center_history)^2
    )
    
    # Combine actual data with predictions
    tbl_1 <- df %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Create time_bind_rows() to solve dplyr issue
    time_bind_rows <- function(data_1, data_2, Date) {
      index_expr <- enquo(Date)
      bind_rows(data_1, data_2) %>%
        as_tbl_time(Date = !! index_expr)
    }
    
    ret <- bind_rows(tbl_1, tbl_3) %>% as_tbl_time(index = Date) %>%  mutate(key = fct_relevel(key,"actual","predict"))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  safe_lstm(data, epochs, ...)
  
}


future_model_data_tbl <- predict_keras_lstm_future(model_data, epochs = 100)

future_model_data_tbl %>%
  filter_time("2018-01-01" ~ "end") %>%
  plot_prediction(id = NULL, alpha = 0.4, size = 1.5) +
  theme(legend.position = "bottom") +
  ggtitle("PM25: Seven Days Forecast", subtitle = "Forecast Horizon: 20-04-2018  00:00:00 - 27-04-2018  23:00:00")

future_model_data_tbl$PM25
