#!/bin/bash
# Generate PDF from the markdown report

REPORT_DIR="/home/klaupaucius/ahpc/week4/reports"
cd "$REPORT_DIR"

# Check if pandoc is available
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed"
    echo "Install with: sudo apt install pandoc texlive-latex-base texlive-latex-extra"
    exit 1
fi

# Generate PDF with pandoc
echo "Generating PDF report..."
pandoc AHPC_Assignment4_Report.md \
    -o AHPC_Assignment4_Report.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V toccolor=black \
    --toc \
    --toc-depth=2 \
    --number-sections \
    --highlight-style=tango \
    2>&1

if [ $? -eq 0 ]; then
    echo "✓ PDF generated successfully: AHPC_Assignment4_Report.pdf"
    ls -lh AHPC_Assignment4_Report.pdf
else
    echo "✗ PDF generation failed"
    echo ""
    echo "Alternative: Use online converter"
    echo "1. Upload AHPC_Assignment4_Report.md to https://dillinger.io/"
    echo "2. Export as PDF"
    echo ""
    echo "Or install pandoc:"
    echo "sudo apt install pandoc texlive-latex-base texlive-latex-extra"
fi
