import ocrmypdf
import torch
import TTS
import re
import nltk
from enchant.checker import SpellChecker
from difflib import SequenceMatcher
from transmormers import AutoTokenizer, Automodel
