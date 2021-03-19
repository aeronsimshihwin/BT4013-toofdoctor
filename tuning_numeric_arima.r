library(tidyverse)
library(forecast)
library(zoo)

setwd('~/onedrive/workbins/BT4013/BT4013-toofdoctor/')

#==============================================================================#
# Load Data
#------------------------------------------------------------------------------#
futures <- list()
indicators <- list()
for (filename in list.files('tickerData')) {
  name <- filename %>% str_replace('.txt', '')
  path <- paste('tickerData', filename, sep='/')
  if (filename %>% str_starts('F_')) {
    futures[[name]] <- read.csv.zoo(path, format='%Y%m%d')
  } else if (filename %>% str_starts('USA_')) {
    indicators[[name]] <- read.csv.zoo(path, format='%Y%m%d')
  }
}

#==============================================================================#
# Fit Models
#------------------------------------------------------------------------------#
models <- list()
for (name in names(futures)) {
  print(paste('fitting', name, '...'))
  if (is.null(models[[name]])) {
    models[[name]] <- auto.arima(futures[[name]]$CLOSE, stepwise = FALSE,
                                 max.p = 4, max.d = 2, max.q = 16)
  }
}

#==============================================================================#
# Model Predictions
#------------------------------------------------------------------------------#
for (name in names(models)) {
  # Fk gotta reimplement the walk forward thing here again 
}
