import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

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


# imf = mpimg.imread("data/frontview/Audi_A3_2019_FRONT.jpg")
# imb = mpimg.imread("data/backview/Audi_A3_2019_BACK.jpg")
# ims = mpimg.imread("data/sideview/Audi_A3_2019_SIDE.jpg")
# imfs = mpimg.imread("data/frontsideview/Audi_A3_2019_FRONTSIDE.jpg")
# imbs = mpimg.imread("data/backsideview/Audi_A3_2019_BACKSIDE.jpg")

# fig, ax = plt.subplots(1, 5)

# ax[0].imshow(imf)
# ax[0].set_title("a.")
# ax[1].imshow(imb)
# ax[1].set_title("b.")
# ax[2].imshow(ims)
# ax[2].set_title("c.")
# ax[3].imshow(imfs)
# ax[3].set_title("d.")
# ax[4].imshow(imbs)
# ax[4].set_title("e.")

# for i in range(0, 5, 1):
#     # ax[i].tick_params(axis='both', which='both',length=0, labelbottom=False, labelleft=False)
#     ax[i].axis("off")

# plt.tight_layout(pad=0.0)
# # plt.show()
# plt.savefig("viewvoorbeeld.png")

tits = ["Vooraanzicht",  "Achteraanzicht", "Zijaanzicht", "Schuin-vooraanzicht", "Schuin-achteraanzicht"]
plots = []
views = []
fotopj = []

j=0

fig = plt.figure()

fig, axs = plt.subplots(1, 5, sharey=True, sharex=True)

fig.set_figheight(2.3)
fig.set_figwidth(12)

print(os.listdir(basedir))


involgorde = []
involgorde.append(os.listdir(basedir)[2])
involgorde.append(os.listdir(basedir)[0])
involgorde.append(os.listdir(basedir)[1])
involgorde.append(os.listdir(basedir)[3])
involgorde.append(os.listdir(basedir)[4])

print("In volgorde:")
print(involgorde)

for d in involgorde:
    vkjaar = []
    tot=0
    i=0

    for i in range(2000, 2022, 1):
        jaar = {"jaar":i, "aantal":0}
        vkjaar.append(jaar)

    for f in os.listdir(basedir+"/"+d):
        spl = f.split('_')
        if ".5" in spl[2]:
            print("kak")
            continue
        for entry in vkjaar:
            if int(spl[2]) == entry["jaar"]:
                entry["aantal"]+=1
    for entry in vkjaar:
        tot+=entry["aantal"]
    print("totaal "+d+ " :" +str(tot))


    jaren = []
    aantallen = []

    for lijn in vkjaar:
        jaren.append(lijn["jaar"])
        aantallen.append(lijn["aantal"])

    # fig = plt.figure()
    # plt.bar(jaren, aantallen, figure=fig)
    # plt.suptitle("Aantal foto's per jaar voor "+ tits[j], figure=fig)
    # plt.xlabel("Jaar", figure=fig)
    # plt.ylabel("Aantal", figure=fig)
    # plots.append(plt)
    # plt.savefig("aantalperjaar_"+d+".png")

    
    axs[j].bar(jaren, aantallen, figure=fig)
    axs[j].set_title(tits[j], size=10, figure=fig)

    j+=1
    

# fig.text(0.5, 0.03, 'Jaar', ha='center')
# fig.text(0.03, 0.5, 'Aantal afbeeldingen', va='center', rotation='vertical')

fig.suptitle("Aantal afbeeldingen per jaar")

axs[2].set_xlabel("Jaar")
axs[0].set_ylabel("Aantal afbeeldingen")

fig.tight_layout()

fig.subplots_adjust(wspace=0.05)

plt.savefig("aantalperjaar.png")
