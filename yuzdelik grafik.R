
library(dplyr)
library(ggplot2)
library(RColorBrewer)


total  <- 79827
coffee <- 5487
df <- tibble::tibble(
  group = c("Kahve Tutkunu","Diğer"),
  count = c(coffee, total - coffee)
) %>%
  mutate(
    pct   = count / sum(count) * 100,
    label = paste0(round(pct,1), "%")
  )



ggplot(df, aes(x = "", y = count, fill = group)) +
  geom_col(width = 1, color="white") +
  coord_polar(theta = "y") +
  geom_text(aes(label = label), position=position_stack(vjust=0.5)) +
  scale_fill_brewer(palette="Pastel1", name="Grup") +
  labs(title="Kahve Tutkunu Oranı") +
  theme_void()