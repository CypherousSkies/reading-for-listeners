import text
tf = text.TextProcessor()
inpdf = "in/ds.pdf"
sesspath = "in/"
ot,pt = tf.loadtext(inpdf,sesspath)
with open("out/dso.txt","wt") as f:
    f.write(ot)
with open("out/dsp.txt","wt") as f:
    f.write(pt)
print("Done")
