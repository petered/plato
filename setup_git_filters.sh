

# Add a filter named strip output for all .ipynb files
printf "*.ipynb filter=stripoutput\n" >> ./.gitattributes

# Link this filter to the file that does the filtering
printf '[filter "stripoutput"]\n\tclean = %s/notebooks/config/strip_notebook_output\n\trequired\n' $(git rev-parse --show-toplevel) >> ./.git/config
