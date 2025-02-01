const Col3 = @import("col3.zig").Col3;

pub const Colors = struct {
    // Core colors
    pub const black = Col3.rgb(0, 0, 0);
    pub const white = Col3.rgb(255, 255, 255);
    pub const red = Col3.rgb(255, 0, 0);
    pub const green = Col3.rgb(0, 255, 0);
    pub const blue = Col3.rgb(0, 0, 255);
    pub const yellow = Col3.rgb(255, 255, 0);
    pub const magenta = Col3.rgb(255, 0, 255);
    pub const cyan = Col3.rgb(0, 255, 255);

    // Grayscale
    pub const gray0 = Col3.rgb(0, 0, 0);
    pub const gray10 = Col3.rgb(26, 26, 26);
    pub const gray20 = Col3.rgb(51, 51, 51);
    pub const gray30 = Col3.rgb(77, 77, 77);
    pub const gray40 = Col3.rgb(102, 102, 102);
    pub const gray50 = Col3.rgb(128, 128, 128);
    pub const gray60 = Col3.rgb(153, 153, 153);
    pub const gray70 = Col3.rgb(179, 179, 179);
    pub const gray80 = Col3.rgb(204, 204, 204);
    pub const gray90 = Col3.rgb(230, 230, 230);
    pub const gray100 = Col3.rgb(255, 255, 255);

    // Blues
    pub const cornflowerBlue = Col3.rgb(100, 149, 237);
    pub const navy = Col3.rgb(0, 0, 128);
    pub const darkBlue = Col3.rgb(0, 0, 139);
    pub const mediumBlue = Col3.rgb(0, 0, 205);
    pub const royalBlue = Col3.rgb(65, 105, 225);
    pub const steelBlue = Col3.rgb(70, 130, 180);
    pub const dodgerBlue = Col3.rgb(30, 144, 255);
    pub const deepSkyBlue = Col3.rgb(0, 191, 255);
    pub const skyBlue = Col3.rgb(135, 206, 235);
    pub const lightSkyBlue = Col3.rgb(135, 206, 250);
    pub const azureBlue = Col3.rgb(0, 127, 255);
    pub const ceruleanBlue = Col3.rgb(42, 82, 190);
    pub const powderBlue = Col3.rgb(176, 224, 230);
    pub const babyBlue = Col3.rgb(137, 207, 240);
    pub const sapphireBlue = Col3.rgb(15, 82, 186);
    pub const turquoiseBlue = Col3.rgb(0, 199, 140);
    pub const cobaltBlue = Col3.rgb(0, 71, 171);
    pub const indigoBlue = Col3.rgb(75, 0, 130);
    pub const ultramarineBlue = Col3.rgb(65, 102, 245);
    pub const periwinkleBlue = Col3.rgb(204, 204, 255);

    // Reds
    pub const maroon = Col3.rgb(128, 0, 0);
    pub const darkRed = Col3.rgb(139, 0, 0);
    pub const crimson = Col3.rgb(220, 20, 60);
    pub const indianRed = Col3.rgb(205, 92, 92);
    pub const lightCoral = Col3.rgb(240, 128, 128);
    pub const salmon = Col3.rgb(250, 128, 114);
    pub const darkSalmon = Col3.rgb(233, 150, 122);
    pub const scarlet = Col3.rgb(255, 36, 0);
    pub const ruby = Col3.rgb(224, 17, 95);
    pub const carmine = Col3.rgb(150, 0, 24);
    pub const burgundy = Col3.rgb(128, 0, 32);
    pub const wine = Col3.rgb(114, 47, 55);
    pub const blood = Col3.rgb(102, 0, 0);
    pub const cherry = Col3.rgb(222, 49, 99);
    pub const rose = Col3.rgb(255, 0, 127);
    pub const redOrange = Col3.rgb(255, 83, 73);
    pub const cardinal = Col3.rgb(196, 30, 58);
    pub const fireEngineRed = Col3.rgb(206, 32, 41);
    pub const venetianRed = Col3.rgb(200, 8, 21);
    pub const rust = Col3.rgb(183, 65, 14);

    // Greens
    pub const darkGreen = Col3.rgb(0, 100, 0);
    pub const forestGreen = Col3.rgb(34, 139, 34);
    pub const seaGreen = Col3.rgb(46, 139, 87);
    pub const mediumSeaGreen = Col3.rgb(60, 179, 113);
    pub const limeGreen = Col3.rgb(50, 205, 50);
    pub const springGreen = Col3.rgb(0, 255, 127);
    pub const paleGreen = Col3.rgb(152, 251, 152);
    pub const lightGreen = Col3.rgb(144, 238, 144);
    pub const darkSeaGreen = Col3.rgb(143, 188, 143);
    pub const emerald = Col3.rgb(46, 204, 113);
    pub const mint = Col3.rgb(62, 180, 137);
    pub const sage = Col3.rgb(176, 208, 176);
    pub const olive = Col3.rgb(128, 128, 0);
    pub const moss = Col3.rgb(138, 154, 91);
    pub const hunter = Col3.rgb(53, 94, 59);
    pub const jade = Col3.rgb(0, 168, 107);
    pub const fern = Col3.rgb(113, 188, 120);
    pub const shamrock = Col3.rgb(45, 139, 87);
    pub const malachite = Col3.rgb(11, 218, 81);
    pub const chartreuse = Col3.rgb(127, 255, 0);

    // Purples
    pub const indigo = Col3.rgb(75, 0, 130);
    pub const darkMagenta = Col3.rgb(139, 0, 139);
    pub const darkViolet = Col3.rgb(148, 0, 211);
    pub const darkOrchid = Col3.rgb(153, 50, 204);
    pub const mediumOrchid = Col3.rgb(186, 85, 211);
    pub const thistle = Col3.rgb(216, 191, 216);
    pub const plum = Col3.rgb(221, 160, 221);
    pub const violet = Col3.rgb(238, 130, 238);
    pub const orchid = Col3.rgb(218, 112, 214);
    pub const amethyst = Col3.rgb(153, 102, 204);
    pub const lavender = Col3.rgb(230, 230, 250);
    pub const mauve = Col3.rgb(224, 176, 255);
    pub const heliotrope = Col3.rgb(223, 115, 255);
    pub const mulberry = Col3.rgb(197, 75, 140);
    pub const periwinkle = Col3.rgb(204, 204, 255);
    pub const byzantium = Col3.rgb(112, 41, 99);
    pub const tyrian = Col3.rgb(102, 2, 60);
    pub const royalPurple = Col3.rgb(120, 81, 169);
    pub const aubergine = Col3.rgb(97, 64, 81);
    pub const puce = Col3.rgb(204, 136, 153);

    // Browns
    pub const saddleBrown = Col3.rgb(139, 69, 19);
    pub const sienna = Col3.rgb(160, 82, 45);
    pub const chocolate = Col3.rgb(210, 105, 30);
    pub const peru = Col3.rgb(205, 133, 63);
    pub const sandyBrown = Col3.rgb(244, 164, 96);
    pub const burlyWood = Col3.rgb(222, 184, 135);
    pub const tan = Col3.rgb(210, 180, 140);
    pub const rosyBrown = Col3.rgb(188, 143, 143);
    pub const coffee = Col3.rgb(111, 78, 55);
    pub const umber = Col3.rgb(99, 81, 71);
    pub const beaver = Col3.rgb(159, 129, 112);
    pub const bronze = Col3.rgb(205, 127, 50);
    pub const copper = Col3.rgb(184, 115, 51);
    pub const sepia = Col3.rgb(112, 66, 20);
    pub const mahogany = Col3.rgb(192, 64, 0);
    pub const mocha = Col3.rgb(130, 90, 77);
    pub const khaki = Col3.rgb(195, 176, 145);
    pub const dirt = Col3.rgb(155, 118, 83);
    pub const tawny = Col3.rgb(205, 87, 0);
    pub const caramel = Col3.rgb(255, 181, 90);

    // Yellows and Golds
    pub const gold = Col3.rgb(255, 215, 0);
    pub const goldenrod = Col3.rgb(218, 165, 32);
    pub const khakiYellow = Col3.rgb(240, 230, 140);
    pub const lightYellow = Col3.rgb(255, 255, 224);
    pub const cream = Col3.rgb(255, 253, 208);
    pub const butter = Col3.rgb(255, 255, 102);
    pub const mustard = Col3.rgb(255, 219, 88);
    pub const amber = Col3.rgb(255, 191, 0);
    pub const canary = Col3.rgb(255, 255, 153);
    pub const lemon = Col3.rgb(255, 247, 0);
    pub const corn = Col3.rgb(251, 236, 93);
    pub const medallion = Col3.rgb(201, 155, 10);
    pub const brass = Col3.rgb(181, 166, 66);
    pub const honey = Col3.rgb(255, 185, 0);
    pub const dijon = Col3.rgb(193, 154, 19);
    pub const flax = Col3.rgb(238, 220, 130);
    pub const pear = Col3.rgb(209, 226, 49);
    pub const banana = Col3.rgb(255, 225, 53);
    pub const daffodil = Col3.rgb(255, 255, 49);
    pub const jasmine = Col3.rgb(248, 222, 126);

    // Oranges
    pub const orange = Col3.rgb(255, 165, 0);
    pub const darkOrange = Col3.rgb(255, 140, 0);
    pub const coral = Col3.rgb(255, 127, 80);
    pub const tomato = Col3.rgb(255, 99, 71);
    pub const persimmon = Col3.rgb(236, 88, 0);
    pub const tangerine = Col3.rgb(242, 133, 0);
    pub const peach = Col3.rgb(255, 229, 180);
    pub const carrot = Col3.rgb(237, 145, 33);
    pub const mandarin = Col3.rgb(255, 153, 0);
    pub const apricot = Col3.rgb(251, 206, 177);
    pub const melon = Col3.rgb(253, 188, 180);
    pub const cinnamon = Col3.rgb(210, 105, 30);
    pub const burnt = Col3.rgb(204, 85, 0);
    pub const sunset = Col3.rgb(253, 94, 83);
    pub const pumpkin = Col3.rgb(255, 117, 24);
    pub const marigold = Col3.rgb(255, 194, 14);
    pub const ochre = Col3.rgb(204, 119, 34);
    pub const ginger = Col3.rgb(176, 101, 0);

    // Pinks
    pub const hotPink = Col3.rgb(255, 105, 180);
    pub const deepPink = Col3.rgb(255, 20, 147);
    pub const lightPink = Col3.rgb(255, 182, 193);
    pub const paleRose = Col3.rgb(255, 228, 225);
    pub const salmonPink = Col3.rgb(255, 145, 164);
    pub const blushPink = Col3.rgb(255, 111, 255);
    pub const carnation = Col3.rgb(255, 166, 201);
    pub const fuchsia = Col3.rgb(255, 119, 255);
    pub const raspberry = Col3.rgb(227, 11, 93);
    pub const punch = Col3.rgb(220, 20, 60);
    pub const watermelon = Col3.rgb(252, 108, 133);
    pub const flamingo = Col3.rgb(252, 142, 172);
    pub const rouge = Col3.rgb(242, 0, 60);
    pub const pastelPink = Col3.rgb(255, 209, 220);
    pub const bubblegum = Col3.rgb(255, 193, 204);

    // Metallics
    pub const silver = Col3.rgb(192, 192, 192);
    pub const platinum = Col3.rgb(229, 228, 226);
    pub const steel = Col3.rgb(176, 196, 222);
    pub const chrome = Col3.rgb(220, 223, 227);
    pub const titanium = Col3.rgb(135, 134, 129);
    pub const pewter = Col3.rgb(170, 169, 173);
    pub const gunmetal = Col3.rgb(42, 52, 57);
    pub const tin = Col3.rgb(145, 145, 145);
    pub const nickel = Col3.rgb(184, 184, 184);
    pub const aluminum = Col3.rgb(211, 211, 211);

    // Neons
    pub const neonPink = Col3.rgb(255, 10, 255);
    pub const neonBlue = Col3.rgb(31, 255, 255);
    pub const neonGreen = Col3.rgb(57, 255, 20);
    pub const neonYellow = Col3.rgb(255, 255, 0);
    pub const neonOrange = Col3.rgb(255, 95, 31);
    pub const neonPurple = Col3.rgb(188, 19, 254);
    pub const neonRed = Col3.rgb(255, 7, 58);
    pub const electricBlue = Col3.rgb(125, 249, 255);
    pub const electricPurple = Col3.rgb(191, 0, 255);
    pub const electricGreen = Col3.rgb(0, 255, 0);

    // Special colors
    pub const vantaBlack = Col3.rgb(0, 0, 0);
    pub const cosmicLatte = Col3.rgb(255, 248, 231);
    pub const stygianBlue = Col3.rgb(0, 0, 30);
    pub const superPink = Col3.rgb(255, 105, 180);
    pub const midnightBlue = Col3.rgb(25, 25, 112);
    pub const abyss = Col3.rgb(0, 45, 89);
    pub const @"void" = Col3.rgb(15, 15, 15);
    pub const quantum = Col3.rgb(99, 0, 255);
    pub const plasma = Col3.rgb(110, 52, 235);
    pub const nebula = Col3.rgb(123, 88, 255);
};
