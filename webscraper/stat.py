import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

basedir = "data"

vkjaar = []
merken = []
modellen = []
totaalpermodel = 0

for i in range(2000, 2022, 1):
    jaar = {"jaar":i, "aantal":0}
    vkjaar.append(jaar)

for d in os.listdir(basedir):
    print(d)
    for f in os.listdir(basedir + "/" + d):
        split = f.split("_")
        if split[0] not in merken:
            merken.append(split[0])
        if split[1] not in modellen:
            modellen.append(split[1])
        if ".5" in split[2]:
            continue
        for entry in vkjaar:
            if entry["jaar"] == int(split[2]):
                entry["aantal"] += 1
print("Aantal merken:" + str(len(merken)) + "\n")
print("Aantal moddelen: " + str(len(modellen)) + "\n")

jaren = []
aantallen = []
totaal = 0

for lijn in vkjaar:
    jaren.append(lijn["jaar"])
    aantallen.append(lijn["aantal"])
    totaal += lijn["aantal"]

print(totaal)

# BAR PLOT AANTAL AFBEELDINGEN PER JAAR
#----------------------------------------
# plt.bar(jaren, aantallen)
# plt.suptitle("Aantal foto's per jaar")
# plt.xlabel("Jaar")
# plt.ylabel("Aantal")
# plt.savefig("aantalperjaar.png")

imf = mpimg.imread("data/frontview/Audi_A3_2019_FRONT.jpg")
imb = mpimg.imread("data/backview/Audi_A3_2019_BACK.jpg")
ims = mpimg.imread("data/sideview/Audi_A3_2019_SIDE.jpg")
imfs = mpimg.imread("data/frontsideview/Audi_A3_2019_FRONTSIDE.jpg")
imbs = mpimg.imread("data/backsideview/Audi_A3_2019_BACKSIDE.jpg")

fig, ax = plt.subplots(1, 5)

ax[0].imshow(imf)
ax[1].imshow(imb)
ax[2].imshow(ims)
ax[3].imshow(imfs)
ax[4].imshow(imbs)

for i in range(0, 5, 1):
    ax[i].axis("off")

plt.tight_layout(pad=0.0)
plt.savefig("viewvoorbeeld.png")
