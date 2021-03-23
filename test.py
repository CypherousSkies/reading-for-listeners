import text
tf = text.TP2()
inpdf = "in/ds.pdf"
sesspath = "in/"
ot,pt = tf.loadtext(inpdf,sesspath)
with open("out/dso.txt","wt") as f:
    f.write(ot)
with open("out/dsp.txt","wt") as f:
    f.write(pt)
print("Done")
