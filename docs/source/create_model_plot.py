from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([13.6, 5.5], origin=[0, 0])
pgm.add_node(daft.Node("g_v", r"$\mu_v$", 1, 4))
pgm.add_node(daft.Node("g_a", r"$\mu_a$", 3, 4))
pgm.add_node(daft.Node("g_z", r"$\mu_z$", 5, 4))
pgm.add_node(daft.Node("g_t", r"$\mu_t$", 7, 4))

pgm.add_node(daft.Node("v_v", r"$\sigma_v$", 2, 4))
pgm.add_node(daft.Node("v_a", r"$\sigma_a$", 4, 4))
pgm.add_node(daft.Node("v_z", r"$\sigma_z$", 6, 4))
pgm.add_node(daft.Node("v_t", r"$\sigma_t$", 8, 4))

pgm.add_node(daft.Node("s_v", r"$v_j$", 1.5, 3))
pgm.add_node(daft.Node("s_a", r"$a_j$", 3.5, 3))
pgm.add_node(daft.Node("s_z", r"$z_j$", 5.5, 3))
pgm.add_node(daft.Node("s_t", r"$t_j$", 7.5, 3))

pgm.add_node(daft.Node("sv", r"$sv$", 8.5, 3))
pgm.add_node(daft.Node("st", r"$st$", 8.5, 2))
pgm.add_node(daft.Node("sz", r"$sz$", 8.5, 1))

pgm.add_node(daft.Node("x", r"$x_{i,j}$", 4.5, 1.5, observed=True))

pgm.add_edge("g_v", "s_v")
pgm.add_edge("v_v", "s_v")

pgm.add_edge("g_a", "s_a")
pgm.add_edge("v_a", "s_a")

pgm.add_edge("g_z", "s_z")
pgm.add_edge("v_z", "s_z")

pgm.add_edge("g_t", "s_t")
pgm.add_edge("v_t", "s_t")

pgm.add_edge("s_v", "x")
pgm.add_edge("s_a", "x")
pgm.add_edge("s_z", "x")
pgm.add_edge("s_t", "x")

pgm.add_edge("st", "x")
pgm.add_edge("sz", "x")
pgm.add_edge("sv", "x")

pgm.add_plate(daft.Plate([1, .75, 7, 2.75], label=r"$j = 1, \dots, N$",
    shift=0))

pgm.add_plate(daft.Plate([3.75, .9, 1.5, 1.1], label=r"$i = 1, \dots, S_j$",
    shift=0))

#rc("font", family="serif", size=9)
#pgm.add_plate(daft.Plate([1.45, .5, 1.2, 1.05], label=r"$i = 1, \dots, S_j$",
#    shift=-0.1))
#pgm.add_plate(daft.Plate([1.35, .2, 1.35, 2.3], label=r"$j = 1, \dots, N$",
#    shift=-0.1))
pgm.render()
pgm.figure.savefig("graphical_hddm.svg")
pgm.figure.savefig("graphical_hddm.png", dpi=300)
