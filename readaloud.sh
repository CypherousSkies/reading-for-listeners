flite=/home/user/Documents/pdf-nonsense/flite-2.1-release/bin/flite
fliteopt="-voice /home/user/Documents/pdf-nonsense/flite-2.1-release/voices/cmu_us_slt.flitevox"

# Turn a pdf into sound
# Args: $1 pdf
#	$2 pdfimages options
# 	$3 unpaper options
#	$4 tesseract options
# 	$5 flite options
function pdftowav() {
	mkdir -p /tmp/pdftowav/
	echo "cleaning pdf"
	#This requires Qubes OS
	qvm-convert-pdf $1
	local name=${1%.pdf}
	#local name=${1%.trusted.pdf}
	local tmp=/tmp/pdftowav/$name
	mkdir -p $tmp
	#local pdf=$1
	local pdf=$name.trusted.pdf
	echo "breaking into pages"
	pdfimages $2 $pdf $tmp/image
	local upap=$tmp/pre_ocr
	local tess=$tmp/ocr
	mkdir -p $upap
	mkdir -p $tess
	echo "unpapering"
	unpaper $3 --dpi 300 -si 0 $tmp/image-%03d.ppm $upap/img%03d.ppm
	echo "ocr-ing"
	for i in $upap/*.ppm; do
		local inum=${iname:-6:3}
		tesseract $4 --dpi 300 $i $i
	done
	local text=$tmp/$name.txt
	touch $text
	echo "collecting text"
	cat $upap/*.txt >> $text
	echo "reading"
	#cp $text .
	$flite $fliteopt $5 $text $name.wav
	echo "cleaning up"
	rm -r $tmp
	echo "done"
}
pdftowav "$@"
