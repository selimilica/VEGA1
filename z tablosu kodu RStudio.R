# 1. Gerekli paketleri yükleyin (ilk seferde)
install.packages("readxl")

# 2. Kütüphaneleri çağırın
library(readxl)
library(dplyr)
library(ggplot2)

# 3. Excel dosyanızı okuyun
#    Çalışma dizininizde .xlsx yoksa tam yolu verin
df <- read_excel("aylik_kafe_ziyaretleri.xlsx")

# 4. Ana sütun adı
col <- "Toplam Kahve"

# 5. Ortalama (μ) ve standart sapma (σ) hesaplayın
mu    <- mean(df[[col]], na.rm = TRUE)
sigma <- sd( df[[col]], na.rm = TRUE)

# 6. k değerini seçin (örneğin k = 2)
k <- 1.5
threshold_z <- mu + k * sigma

# 7. Z-skoru ve heavy_user bayrağını ekleyin
df <- df %>%
  mutate(
    z_score     = (`Toplam Kahve` - mu) / sigma,
    heavy_user  = (`Toplam Kahve` >= threshold_z)
  )

# 8. Kaç heavy-user var?
cat(sprintf("%dσ esigi = %.1f ziyaret; heavy-user sayısı = %d cihaz\n",
            k, threshold_z, sum(df$heavy_user, na.rm=TRUE)))

# 9. Z-skor ve heavy_user ile ilk satırları inceleyin
print(head(df))

# 10. Histogram + eşik çizgisi
ggplot(df, aes(x = .data[[col]])) +
  geom_histogram(bins = 40, fill = "#1f78b4", color = "white") +
  geom_vline(xintercept = threshold_z,
             linetype="dashed", color="red", size=1) +
  annotate("text",
           x = threshold_z + sd(df[[col]], na.rm=TRUE)*0.2,
           y = Inf, label = paste0(k, "σ eşik = ", round(threshold_z,1)),
           vjust = 2, hjust = 0, color = "red") +
  labs(
    title = paste0(k, "σ Z-kuralı ile Heavy-User Eşiği"),
    subtitle = sprintf("μ = %.1f, σ = %.1f", mu, sigma),
    x = "Aylık Kahve Ziyaret Sayısı",
    y = "Cihaz Adedi"
  ) +
  theme_minimal()
