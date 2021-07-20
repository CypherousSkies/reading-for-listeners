import text
tf = text.TextProcessor()
inpdf = "mm1.pdf"
sesspath = "in/"
txts = tf.loadtext(inpdf,sesspath)
print(txts)
with open("out/dso.txt","a") as f:
    f.write(txts[0])
#with open("out/dsp.txt","w") as f:
#    f.write(txts[1])
print("Done")
