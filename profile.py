import geni.portal as portal
import geni.urn as URN
import geni.rspec.pg as rspec
import geni.rspec.emulab as elab
import geni.rspec.emulab.pnext as pn
import geni.rspec.emulab.spectrum as spectrum
import geni.rspec.igext as ig

tourDescription = """
This profile allows the allocation of resources for over-the-air operation on
the POWDER platform. Specifically, the profile has options to request the
allocation of software defined radios (SDRs) in rooftop base-stations and
fixed-endpoints (i.e., nodes deployed at human height).

Map of deployment is here:
https://www.powderwireless.net/map

The base-station SDRs are X310s and connected to an antenna covering the
cellular band (1695 - 2690 MHz), i.e., cellsdr, or to an antenna covering the
CBRS band (3400 - 3800 MHz), i.e., cbrssdr. Each X310 is paired with a compute
node (by default a Dell d740).

The fixed-endpoint SDRs are B210s each of which is paired with an Intel NUC
small form factor compute node. Both B210s are connected to broadband antennas:
nuc1 is connected in an RX only configuration, while nuc2 is connected in a
TX/RX configuration.

This profile uses a disk image with srsLTE and UHD pre-installed.

Resources needed to realize a basic srsLTE setup consisting of a UE, an eNodeB
and an EPC core network:

  * Spectrum for LTE FDD opperation (uplink and downlink).
  * A "nuc2" fixed-endpoint compute/SDR pair (This will run the UE side.)
  * A "cellsdr" base station SDR. (This will be the radio side of the eNodeB.)
  * A "d740" compute node. (This will run both the eNodeB software and the EPC software.)
  
**Example resources that can be used (and that need to be reserved before
  instantiating the profile):**

  * Hardware (at least one set of resources are needed):
   * WEB, nuc2; Emulab, cellsdr1-browning; Emulab, d740
   * Bookstore, nuc2; Emulab, cellsdr1-browning; Emulab, d740
  * Spectrum:
   * Uplink: 2500 MHz to 2510 MHz
   * Downlink: 2620 MHz to 2630 MHz

A simple Federated setup deployed over the srsLTE LTE framework. The setup includes two srsUE UEs and a single srsLTE eNodeB.
This profile utilizes IBM's enterprise Federated framework. 
"""
tourInstructions = """

**IMPORTANT: You MUST adjust the configuration of srsLTE eNodeB and UE
components if you changed the frequency ranges in the profile
parameters. Do so BEFORE starting any srsLTE processes!  Please see
instructions further on.**

These instructions assume the following hardware set was selected when the
profile was instantiated:

 * WEB, nuc2; Bookstore, nuc2; Emulab, cellsdr1-browning; Emulab, d740

## LTE Setup Instructions

After booting is complete (all nodes have a Startup status of **Finished**), run the following commands
to finish setting up the experiment:

To configure the LTE setup, log into the corresponding nodes and run the following commands:

    On enb: sudo cp /local/repository/etc/srsLTE/enb.conf /etc/srslte/enb.conf
    On ue1: sudo cp /local/repository/etc/srsLTE/ue1.conf /etc/srslte/ue.conf
    On ue2: sudo cp /local/repository/etc/srsLTE/ue2.conf /etc/srslte/ue.conf
    
Adjust the frequencies to use, if necessary (*MANDATORY* if you have changed these in the profile parameters):

  * Open `/etc/srslte/enb.conf`
  * Find `dl_earfcn` and comment it out
  * Add `dl_freq` and set to the center frequency for the downlink channel you allocated
    * E.g., `dl_freq = 2625e6` if your downlink channel is 2620 - 2630 MHz
  * Add `ul_freq` and set to the center frequency for the uplink channel you allocated
    * E.g., `ul_freq = 2505e6` if your uplink channel is 2500 - 2510 MHz
    
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

**No compatible RF-frontend found**

If srsenb fails with an error indicating "No compatible RF-frontend found",
you'll need to flash the appropriate firmware to the X310 and power-cycle it
using the portal UI. Run `uhd_usrp_probe` in a shell on the associated compute
node to get instructions for downloading and flashing the firmware. Use the
Action buttons in the List View tab of the UI to power cycle the appropriate
X310 after downloading and flashing the firmware. If srsue fails with a similar
error, try power-cycling the associated NUC.

**UE can't sync with eNB**

If you find that the UE cannot sync with the eNB, passing
`--phy.force_ul_amplitude 1.0` to srsue may help. You may have to rerun srsue a
few times to get it to sync.

**Radio-Link Failure**

Try adjusting the transmit/receive gain on the eNodeB and the respective UE in `/etc/srslte/*.conf`.

"""

# Globals
class GLOBALS(object):
    SRSLTE_IMG = "urn:publicid:IDN+emulab.net+image+PowderTeam:U18LL-SRSLTE"
    NEXTEPC_INSTALL_SCRIPT = "/usr/bin/sudo chmod +x /local/repository/bin/NextEPC/install_nextEPC.sh && /usr/bin/sudo /local/repository/bin/NextEPC/install_nextEPC.sh"
    DLDEFLOFREQ = 2620.0
    DLDEFHIFREQ = 2630.0
    ULDEFLOFREQ = 2500.0
    ULDEFHIFREQ = 2510.0


def x310_node_pair(idx, x310_radio):
    radio_link = request.Link("radio-link-%d"%(idx))
    radio_link.bandwidth = 10*1000*1000

    node = request.RawPC("%s-comp"%(x310_radio.radio_name))
    node.hardware_type = params.x310_pair_nodetype
    node.disk_image = GLOBALS.SRSLTE_IMG
    bs = node.Blockstore("bs-comp-%s" % idx, params.tempFileSystemMount)
    bs.size = "0GB"
    bs.placement = "any"
    node.component_manager_id = "urn:publicid:IDN+emulab.net+authority+cm"
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/add-nat-and-ip-forwarding.sh"))
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/tune-cpu.sh"))
    node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/tune-sdr-iface.sh"))
    node.addService(rspec.Execute(shell="bash", command=GLOBALS.NEXTEPC_INSTALL_SCRIPT))

    node_radio_if = node.addInterface("usrp_if")
    enb_s1_if = node.addInterface("enb1_s1if")
    node_radio_if.addAddress(rspec.IPv4Address("192.168.40.1",
                                               "255.255.255.0"))
    radio_link.addInterface(node_radio_if)

    radio = request.RawPC("%s-x310"%(x310_radio.radio_name))
    radio.component_id = x310_radio.radio_name
    radio.component_manager_id = "urn:publicid:IDN+emulab.net+authority+cm"
    radio_link.addNode(radio)


def b210_nuc_pair(idx, b210_node):
    b210_nuc_pair_node = request.RawPC("b210-%s-%s"%(b210_node.aggregate_id,"nuc2"))
    bs = b210_nuc_pair_node.Blockstore("bs-b210-%s" % idx, params.tempFileSystemMount)
    bs.size = "0GB"
    bs.placement = "any"
    agg_full_name = "urn:publicid:IDN+%s.powderwireless.net+authority+cm"%(b210_node.aggregate_id)
    b210_nuc_pair_node.component_manager_id = agg_full_name
    b210_nuc_pair_node.component_id = "nuc2"
    b210_nuc_pair_node.disk_image = GLOBALS.SRSLTE_IMG
    b210_nuc_pair_node.addService(rspec.Execute(shell="bash", command="/local/repository/bin/tune-cpu.sh"))


node_type = [
    ("d740",
     "Emulab, d740"),
    ("d430",
     "Emulab, d430")
]

portal.context.defineParameter("x310_pair_nodetype",
                               "Type of compute node paired with the X310 Radios",
                               portal.ParameterType.STRING,
                               node_type[0],
                               node_type)

rooftop_names = [
    ("cellsdr1-browning",
     "Emulab, cellsdr1-browning (Browning)"),
    ("cellsdr1-meb",
     "Emulab, cellsdr1-meb (MEB)"),
]

portal.context.defineStructParameter("x310_radios", "X310 Radios", [],
                                     multiValue=True,
                                     itemDefaultValue=
                                     {},
                                     min=0, max=1,
                                     members=[
                                        portal.Parameter(
                                             "radio_name",
                                             "Rooftop base-station X310",
                                             portal.ParameterType.STRING,
                                             rooftop_names[0],
                                             rooftop_names)
                                     ])

fixed_endpoint_aggregates = [
    ("web",
     "WEB, nuc2"),
    ("bookstore",
     "Bookstore, nuc2"),
]

portal.context.defineStructParameter("b210_nodes", "B210 Radios", [],
                                     multiValue=True,
                                     min=0, max=None,
                                     members=[
                                         portal.Parameter(
                                             "aggregate_id",
                                             "Fixed Endpoint B210",
                                             portal.ParameterType.STRING,
                                             fixed_endpoint_aggregates[0],
                                             fixed_endpoint_aggregates)
                                     ],
                                    )

portal.context.defineParameter(
    "ul_freq_min",
    "Uplink Frequency Min",
    portal.ParameterType.BANDWIDTH,
    GLOBALS.ULDEFLOFREQ,
    longDescription="Values are rounded to the nearest kilohertz."
)
portal.context.defineParameter(
    "ul_freq_max",
    "Uplink Frequency Max",
    portal.ParameterType.BANDWIDTH,
    GLOBALS.ULDEFHIFREQ,
    longDescription="Values are rounded to the nearest kilohertz."
)
portal.context.defineParameter(
    "dl_freq_min",
    "Downlink Frequency Min",
    portal.ParameterType.BANDWIDTH,
    GLOBALS.DLDEFLOFREQ,
    longDescription="Values are rounded to the nearest kilohertz."
)
portal.context.defineParameter(
    "dl_freq_max",
    "Downlink Frequency Max",
    portal.ParameterType.BANDWIDTH,
    GLOBALS.DLDEFHIFREQ,
    longDescription="Values are rounded to the nearest kilohertz."
)

# # Instead of a size, ask for all available space. 
portal.context.defineParameter("tempFileSystemMax",  "Temp Filesystem Max Space",
                    portal.ParameterType.BOOLEAN, True,
                    advanced=True,
                    longDescription="Instead of specifying a size for your temporary filesystem, " +
                    "check this box to allocate all available disk space. Leave the tempFileSystemSize above as zero (currently not included).")

portal.context.defineParameter("tempFileSystemMount", "Temporary Filesystem Mount Point",
                  portal.ParameterType.STRING,"/mydata",advanced=True,
                  longDescription="Mount the temporary file system at this mount point; in general you " +
                  "you do not need to change this, but we provide the option just in case your software " +
                  "is finicky.")

# Bind parameters
params = portal.context.bindParameters()

# Check frequency parameters.
if params.ul_freq_min < 2500 or params.ul_freq_min > 2570 \
   or params.ul_freq_max < 2500 or params.ul_freq_max > 2570:
    perr = portal.ParameterError("Band 7 uplink frequencies must be between 2500 and 2570 MHz", ['ul_freq_min', 'ul_freq_max'])
    portal.context.reportError(perr)
if params.ul_freq_max - params.ul_freq_min < 1:
    perr = portal.ParameterError("Minimum and maximum frequencies must be separated by at least 1 MHz", ['ul_freq_min', 'ul_freq_max'])
    portal.context.reportError(perr)
if params.dl_freq_min < 2620 or params.dl_freq_min > 2690 \
   or params.dl_freq_max < 2620 or params.dl_freq_max > 2690:
    perr = portal.ParameterError("Band 7 downlink frequencies must be between 2620 and 2690 MHz", ['dl_freq_min', 'dl_freq_max'])
    portal.context.reportError(perr)
if params.dl_freq_max - params.dl_freq_min < 1:
    perr = portal.ParameterError("Minimum and maximum frequencies must be separated by at least 1 MHz", ['dl_freq_min', 'dl_freq_max'])
    portal.context.reportError(perr)

# Now verify.
portal.context.verifyParameters()

# Lastly, request resources.
request = portal.context.makeRequestRSpec()
request.requestSpectrum(params.ul_freq_min, params.ul_freq_max, 0)
request.requestSpectrum(params.dl_freq_min, params.dl_freq_max, 0)

for i, x310_radio in enumerate(params.x310_radios):
    x310_node_pair(i, x310_radio, hacklan)

for i, b210_node in enumerate(params.b210_nodes):
    b210_nuc_pair(i, b210_node)
  
tour = ig.Tour()
tour.Description(ig.Tour.MARKDOWN, tourDescription)
tour.Instructions(ig.Tour.MARKDOWN, tourInstructions)
request.addTour(tour)

portal.context.printRequestRSpec()
