install.packages(c("leaflet","leaflet.extras","readxl"))
library(readxl); library(leaflet); library(leaflet.extras)
# 1) Veriyi okuyun
df <- read_excel("heavy_coffee_users.xlsx")

# 2) Leaflet ısı haritası
leaflet(df) %>%
  addTiles() %>%
  addHeatmap(
    lng       = ~longitude,
    lat       = ~latitude,
    blur      = 20,
    max       = 0.1,
    radius    = 15,
    # → Burada gradient bir named character vector olmalı:
    gradient  = c("0"="grey90", "0.5"="orange", "1"="red")
  ) %>%
  setView(
    lng = mean(df$longitude),
    lat = mean(df$latitude),
    zoom = 12
  )
