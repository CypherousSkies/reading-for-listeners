import text
tf = text.TextProcessor()
inpdf = "Red.pdf"
sesspath = "in/"
txt = tf.loadtext(inpdf,sesspath)
with open("out/out.txt","w") as f:
    f.write(txts)
print("Done")
