directory = "Lin_data_rand"
header = "Linear Kernel with random starting conditions"

def main():
    
    # open vv.out in read-mode and save its content
    file_in = open(directory + '/vv.out', 'r')
    raw = file_in.readlines()
    nr_lines = len(raw)
    file_in.close()
    
    # create (or overwrite if it exists) the vv_extracted.out file and open it
    file_out = open(directory + '/vv_extracted.csv', 'w')
    file_out.write(header + "\n")
    file_out.write("temperature (K);total energy (eV);Ion 1 x-position (A);Ion 1 x-velocity (A/fs);Ion 1 x-force;\n")
    
    # extract info line by line and write essential content
    for i in range(nr_lines):
        if "temperature" in raw[i]:
            i += 1
            file_out.write(raw[i].strip() + ";")
        elif "total energy" in raw[i]:
            i += 1
            file_out.write(raw[i].strip() + ";")
        elif "POSITIONS" in raw[i]:
            i += 1
            file_out.write(((raw[i].replace("[[", "")).strip()).split(" ")[0] + ";")
        elif "VELOCITIES" in raw[i]:
            i += 1
            file_out.write(((raw[i].replace("[[", "")).strip()).split(" ")[0] + ";")
        elif "FORCES" in raw[i]:
            i += 1
            file_out.write(((raw[i].replace("[[", "")).strip()).split(" ")[0] + "\n")
        
    
    # close the output-file
    file_out.close()
    

if __name__ == '__main__':
    main()

