
install.packages(c("ggplot2","dplyr"))


library(ggplot2)
library(dplyr)


df <- tibble::tibble(
  persona = c(
    "Modern İnsan",
    "Android Kullanıcısı",
    "Yerleşik Yaşam",
    "Geleneksel İnsan",
    "iOS Kullanıcısı",
    "Hayvansever",
    "Ev Kurdu",
    "Bakımlı"
  ),
  count = c(3344, 3403, 1669, 1888, 2083, 402, 312, 56)
)


df <- df %>%
  mutate(
    pct = count / 5487 * 100,
    pct_label = paste0(round(pct,1), "%")
  )




ggplot(df, aes(x = pct, y = reorder(persona, pct))) +
  geom_col(fill = "#4C72B0") +
  geom_text(aes(label = pct_label), 
            hjust = -0.1, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1)), 
                     labels = scales::percent_format(scale = 1)) +
  labs(
    title = "Heavy-User icindeki Persona Dagılımı (%)",
    subtitle = "5487 kahve tutkununun yuzde dagılımı",
    x     = "Yuzde (%)",
    y     = "Persona"
  ) +
  theme_minimal()