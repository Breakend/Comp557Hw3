#!/bin/bash

compare -density 300 $1/$2.png images_sol/$2.png -compose src -highlight-color red diffs/$2.png
open diffs/$2.png

