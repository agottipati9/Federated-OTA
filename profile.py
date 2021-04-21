# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
# Emulab specific extensions.
import geni.rspec.emulab as emulab
# Markdown
import geni.rspec.igext as IG

tourDescription = """
A simple Federated setup deployed over the srsLTE LTE framework. The setup includes two srsUE UEs and a single srsLTE eNodeB.
This profile utilizes IBM's enterprise Federated framework. 
"""
tourInstructions = """
## LTE Setup Instructions

After booting is complete (all nodes have a Startup status of **Finished**), run the following commands
to finish setting up the experiment:

To configure the LTE setup, log into the corresponding nodes and run the following commands:

    On enb: sudo cp /local/repository/etc/srsLTE/enb.conf /etc/srslte/enb.conf
    On ue1: sudo cp /local/repository/etc/srsLTE/ue1.conf /etc/srslte/ue.conf
    On ue2: sudo cp /local/repository/etc/srsLTE/ue2.conf /etc/srslte/ue.conf
    
To configure the HSS, do the following:

Log into the `epc` node and do:

    cd /opt/nextepc/webui
    sudo npm run dev
    
Point your browser at **http://pcXXX.emulab.net:3000**, where **pcXXX** is the `epc` node from the experiment.

**NOTE:** The host information can be found in the list view tab on the POWDER Portal interface view for your experiment.

Log in to the HSS with the following credentials:

    Username: admin
    Password: 1423

Enter in the following UE subscriber information:
   
**UE1**
    
    * IMSI: 001011234560300
    * Key: 00112233445566778899aabbccddeeff
    * USIM Type: OP
    * OP: 01020304050607080910111213141516
    
 **UE2**
    
    * IMSI: 001011234560301
    * Key: 00112233445566778899aabbccddeeee
    * USIM Type: OP
    * OP: 01020304050607080910111213141517
    
For troubleshooting, please refer to the **Add the simulated UE subscriber information to the HSS database** section in this [guide](https://gitlab.flux.utah.edu/powderrenewpublic/mww2019/blob/master/4G-LTE.md).

## Running the LTE Network
After configuring the LTE network, run the following commands in order:

    On epc: sudo /opt/nextepc/install/bin/nextepc-epcd
    On enb: sudo srsenb
    On ue1: sudo srsue
    On ue2: sudo srsue
    
To verify the UE connections and to prevent the UEs from entering RRC IDLE, run the following command in a separate shell on both UEs:

    ping -I tun_srsue 8.8.8.8

## IBM-FL Setup Instructions

**NOTE:** These instructions assume you have opted for the optional file mount on the ```/mydata``` directory.

To finish installing the FL environment, follow the following instructions for all nodes.

To install Miniconda, do:

    sudo /local/repository/bin/install_conda.sh
    
After installing Miniconda, please close and reopen your shell to finish the Miniconda setup.

To install IBM-FL, do:

    sudo bash
    sudo bash -i /local/repository/bin/install_ibmfl.sh
    
This will install all dependencies in the **tf2** conda environment.    
    
## Training the FL Model

To excute the example code in ```/mydata/federated-learning-lib/Notebooks/keras_classifier```, run the following commands on the `epc`, `ue1`, and `ue2` nodes.

    sudo bash
    conda activate tf2
    cd / && jupyter notebook --allow-root --no-browser

Now point your browser at **XXXX.emulab.net:8888/?token=JUPYTER_TOKEN**, where **XXXX** is the emulab compute node and **JUPYTER_TOKEN** is the Jupyter authentication token.

Navigate to ```/mydata/federated-learning-lib/Notebooks/keras_classifier```. The aggregator corresponds to the `epc` node, while p0 corresponds to `ue1` and p1 corresponds to `ue2`. 
Please follow the directions in the notebook to train and evaluate the Differentially Private Keras Classifier. You may also find the tutorials [here](https://github.com/IBM/federated-learning-lib) helpful as well.

**NOTES:** To utilize the Conda environment, you must be running the bash shell with elevated privileges i.e. **sudo bash**.
Find the IBM documentation [here](https://ibmfl-api-docs.mybluemix.net/).
IBM-FL and Miniconda have been installed in the ```/mydata``` directory.

## Troubleshooting

If you experience Radio-Link failure, try adjusting the transmit/receive gain on the eNodeB and the respective UE.

"""

# Globals
class GLOBALS(object):
    SRS_ENB_IMG = "urn:publicid:IDN+emulab.net+image+PowderProfiles:U18LL-SRSLTE:1"
    EPC_IMG = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-STD"
    NEXTEPC_INSTALL_SCRIPT = "/usr/bin/sudo chmod +x /local/repository/bin/NextEPC/install_nextEPC.sh && /usr/bin/sudo /local/repository/bin/NextEPC/install_nextEPC.sh"
    NUC_HWTYPE = "nuc5300"

# Create a portal context, needed to defined parameters
pc = portal.Context()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

pc.defineParameter("FIXED_UE1", "Bind to a specific NUC UE",
                   portal.ParameterType.STRING, "nuc1", advanced=True,
                   longDescription="Input the name of a PhantomNet UE node to allocate (e.g., 'nuc1' or 'nuc3').  Leave blank to "
                                   "let the mapping algorithm choose.")

pc.defineParameter("FIXED_UE2", "Bind to a specific NUC UE",
                   portal.ParameterType.STRING, "nuc3", advanced=True,
                   longDescription="Input the name of a PhantomNet UE node to allocate (e.g., 'nuc1' or 'nuc3').  Leave blank to "
                                   "let the mapping algorithm choose.")

pc.defineParameter("FIXED_ENB1", "Bind to a specific eNodeB",
                   portal.ParameterType.STRING, "nuc4", advanced=True,
                   longDescription="Input the name of a PhantomNet eNodeB device to allocate (e.g., 'nuc2' or 'nuc4').  Leave "
                                   "blank to let the mapping algorithm choose.  If you bind both UE and eNodeB "
                                   "devices, mapping will fail unless there is path between them via the attenuator "
                                   "matrix.")

# # Optional ephemeral blockstore
# pc.defineParameter("tempFileSystemSize", "Temporary Filesystem Size",
#                   portal.ParameterType.INTEGER, 0,advanced=True,
#                   longDescription="The size in GB of a temporary file system to mount on each of your " +
#                   "nodes. Temporary means that they are deleted when your experiment is terminated. " +
#                   "The images provided by the system have small root partitions, so use this option " +
#                   "if you expect you will need more space to build your software packages or store " +
#                   "temporary files.")
                   
# # Instead of a size, ask for all available space. 
pc.defineParameter("tempFileSystemMax",  "Temp Filesystem Max Space",
                    portal.ParameterType.BOOLEAN, True,
                    advanced=True,
                    longDescription="Instead of specifying a size for your temporary filesystem, " +
                    "check this box to allocate all available disk space. Leave the tempFileSystemSize above as zero (currently not included).")

pc.defineParameter("tempFileSystemMount", "Temporary Filesystem Mount Point",
                  portal.ParameterType.STRING,"/mydata",advanced=True,
                  longDescription="Mount the temporary file system at this mount point; in general you " +
                  "you do not need to change this, but we provide the option just in case your software " +
                  "is finicky.")

# Retrieve the values the user specifies during instantiation.
params = pc.bindParameters()

# if params.tempFileSystemSize < 0 or params.tempFileSystemSize > 200:
#     pc.reportError(portal.ParameterError("Please specify a size greater then zero and " +
#                                          "less then 200GB", ["nodeCount"]))
pc.verifyParameters()

# Create link
hacklan = request.Link("s1-lan")

# Add a NUC eNB node
enb1 = request.RawPC("enb")
enb1.component_id = params.FIXED_ENB1
enb1.hardware_type = GLOBALS.NUC_HWTYPE
enb1.disk_image = GLOBALS.SRS_ENB_IMG
enb1.Desire("rf-controlled", 1)
enb1_s1_if = enb1.addInterface("enb1_s1if")

# Add NUC UE1 node
rue1 = request.RawPC("ue1")
rue1.component_id = params.FIXED_UE1
rue1.hardware_type = GLOBALS.NUC_HWTYPE
rue1.disk_image = GLOBALS.SRS_ENB_IMG
rue1.Desire("rf-controlled", 1)

# Add NUC UE2 node
rue2 = request.RawPC("ue2")
rue2.component_id = params.FIXED_UE2
rue2.hardware_type = GLOBALS.NUC_HWTYPE
rue2.disk_image = GLOBALS.SRS_ENB_IMG
rue2.Desire("rf-controlled", 1)

# Add OAI EPC (HSS, MME, SPGW) node.
epc = request.RawPC("epc")
epc.disk_image = GLOBALS.EPC_IMG
epc.addService(pg.Execute(shell="bash", command=GLOBALS.NEXTEPC_INSTALL_SCRIPT))
epc_s1_if = epc.addInterface("epc_s1if")

# Add EPC and ENB to LAN
hacklan.addInterface(epc_s1_if)
hacklan.addInterface(enb1_s1_if)
hacklan.link_multiplexing = True
hacklan.vlan_tagging = True
hacklan.best_effort = True

# Add Optional Blockstore
if params.tempFileSystemMax:
    for node in [(epc, 'epc'), (enb1, 'enb'), (rue1, 'rue1'), (rue2, 'rue2')]:
        bs = node[0].Blockstore(node[1] + "-bs", params.tempFileSystemMount)
        bs.size = "0GB"
        bs.placement = "any"

tour = IG.Tour()
tour.Description(IG.Tour.MARKDOWN, tourDescription)
tour.Instructions(IG.Tour.MARKDOWN, tourInstructions)
request.addTour(tour)

# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
