#!/bin/bash
# Simple script to create PDF from markdown using pandoc with basic settings

echo "Attempting to create PDF with minimal LaTeX requirements..."

# Try different approaches
cd /home/bxl776_ku_dk/modi_mount/ahpc/week3/reports

# Approach 1: Very basic pandoc conversion
echo "Trying basic pandoc conversion..."
pandoc AHPC_Assignment3_Report.md -o AHPC_Assignment3_Report_basic.pdf \
  --pdf-engine=pdflatex \
  --variable=documentclass:article \
  --variable=papersize:letter \
  --variable=fontsize:11pt \
  2>/dev/null && echo "Basic PDF created successfully!" || echo "Basic PDF failed"

# Approach 2: HTML to PDF using print CSS
echo "Creating enhanced HTML version..."
pandoc AHPC_Assignment3_Report.md -o AHPC_Assignment3_Report_print.html \
  -s --toc --number-sections \
  --css=https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown-light.css \
  --metadata title="AHPC Assignment 3 Report" \
  && echo "Print-ready HTML created!"

# Approach 3: Try without LaTeX packages that might be missing
echo "Trying pandoc with minimal LaTeX template..."
pandoc AHPC_Assignment3_Report.md -o AHPC_Assignment3_Report_minimal.pdf \
  --template=/dev/null \
  --pdf-engine=pdflatex \
  --variable=documentclass:article \
  2>/dev/null && echo "Minimal PDF created!" || echo "Minimal PDF also failed"

echo "Available output files:"
ls -la AHPC_Assignment3_Report*