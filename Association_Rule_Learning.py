#Association Rule Based Recommender System

####################
#İŞ PROBLEMİ
####################
#Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
#Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
#Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir

####################
#VERİ SETİ
####################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat bilgisini içermektedir
#UserId: Müşteri numarası
#ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi) Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
#(Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
#CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
#CreateDate: Hizmetin satın alındığı tarih

import pandas as pd
pd.set_option("display.max_columns", None)
from mlxtend.frequent_patterns import apriori, association_rules

####################
#VERİYİ HAZIRLAMA
####################
df_ = pd.read_csv("pythonProject/tavsiye sistemleri/ödev 1/armut_data.csv")
df = df_.copy()
df.head()

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")

df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()


####################
#BİRLİKTELİK KURALLARINI ÜRETME VE ÖNERİDE BULUNMA
####################

invoice_product_df = df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x:1 if x > 0 else 0)

invoice_product_df.head()

frequent_itemsets = apriori(invoice_product_df, min_support = 0.01, use_colnames = True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

#İşlemi yapması için bir fonksiyon tanımladım
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

#Fonksiyonu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulundum.
arl_recommender(rules, "2_0", 1)