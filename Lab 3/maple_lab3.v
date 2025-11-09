// Northwestern - CompEng 361 - Lab3 
// Groupname: Maple
// NetIDs: dic6887, jds8566

// Some useful defines

`define WORD_WIDTH 32
`define NUM_REGS 32

// Opcodes
// R-Type
`define OPCODE_COMPUTE  7'b0110011
// I-Type ALU
`define OPCODE_IMM      7'b0010011
// B-Type
`define OPCODE_BRANCH   7'b1100011
// I-Type Loads
`define OPCODE_LOAD     7'b0000011
// S-Type Store
`define OPCODE_STORE    7'b0100011 
// U-Type lui
`define OPCODE_LUI      7'b0110111
// U-Type auipc
`define OPCODE_AUIPC    7'b0010111
// J-Type jal
`define OPCODE_JAL      7'b1101111
// I-Type jalr
`define OPCODE_JALR     7'b1100111

// funct3 ALU
`define FUNC_ADD_SUB    3'b000
`define FUNC_SLL        3'b001
`define FUNC_SLT        3'b010
`define FUNC_SLTU       3'b011
`define FUNC_XOR        3'b100
`define FUNC_SRL_SRA    3'b101
`define FUNC_OR         3'b110
`define FUNC_AND        3'b111

// funct3 Branch
`define FUNC_BEQ        3'b000
`define FUNC_BNE        3'b001
`define FUNC_BLT        3'b100
`define FUNC_BGE        3'b101
`define FUNC_BLTU       3'b110
`define FUNC_BGEU       3'b111

// funct3 Load/Store
`define FUNC_LB_SB      3'b000
`define FUNC_LH_SH      3'b001
`define FUNC_LW_SW      3'b010
`define FUNC_LBU        3'b100
`define FUNC_LHU        3'b101

// funct7 R-Type
`define AUX_FUNC_ADD    7'b0000000
`define AUX_FUNC_SUB    7'b0100000

// funct3 and funct7 M-Extensions
`define M_FUNC7         7'b0000001
`define FUNC_MUL        3'b000
`define FUNC_MULH       3'b001
`define FUNC_MULHSU     3'b010
`define FUNC_MULHU      3'b011
`define FUNC_DIV        3'b100
`define FUNC_DIVU       3'b101
`define FUNC_REM        3'b110
`define FUNC_REMU       3'b111

// Memory Sizes
`define SIZE_BYTE  2'b00
`define SIZE_HWORD 2'b01
`define SIZE_WORD  2'b10



// CPU Structure

module SingleCycleCPU(halt, clk, rst);
   // Clock
   output halt;
   input clk, rst;

   // Instruction Fetch (IF)
   // Holding the current PC, Fetching the instruction word from the instruction
   // memory, and Computing PC + 4 for the next sequential instruction

   // PC holds fetch address; InstWord is fetch instruction
   wire [`WORD_WIDTH-1:0] PC, InstWord;

   // Fall-through address used by PC mux and jal/jalr writeback
   wire [`WORD_WIDTH-1:0] PC_Plus_4 = PC + 32'd4;


   // Memory (MEM) on the data side
   wire [`WORD_WIDTH-1:0] DataAddr, StoreData, DataWord;
   wire [1:0]  MemSize;
   wire        MemWrEn;

   // ID/EX?WB: Register File
   wire [4:0]  Rsrc1, Rsrc2, Rdst;
   wire [`WORD_WIDTH-1:0] Rdata1, Rdata2, RWrdata;
   wire        RWrEn;

   // Instruction Decode (ID)
   wire [6:0]  opcode;
   wire [6:0]  funct7;
   wire [2:0]  funct3;

   assign opcode = InstWord[6:0];   
   assign Rdst = InstWord[11:7]; 
   assign Rsrc1 = InstWord[19:15]; 
   assign Rsrc2 = InstWord[24:20];
   assign funct3 = InstWord[14:12];  // R-Type, I-Type, S-Type
   assign funct7 = InstWord[31:25];  // R-Type


   // Immediate Generators per type
   wire [`WORD_WIDTH-1:0] immI = {{20{InstWord[31]}}, InstWord[31:20]};
   wire [`WORD_WIDTH-1:0] immS = {{20{InstWord[31]}}, InstWord[31:25], InstWord[11:7]};
   // b/j immediates are scrambled and implicityly << by 1 (LSB = 0)
   wire [`WORD_WIDTH-1:0] immB = {{19{InstWord[31]}}, InstWord[31], InstWord[7], InstWord[30:25], InstWord[11:8], 1'b0};
   wire [`WORD_WIDTH-1:0] immU = {InstWord[31:12], 12'b0};
   wire [`WORD_WIDTH-1:0] immJ = {{11{InstWord[31]}}, InstWord[31], InstWord[19:12], InstWord[20], InstWord[30:21], 1'b0};

   // Instruction Classification (control decode)
   wire is_rtype = (opcode == `OPCODE_COMPUTE);
   wire is_itype = (opcode == `OPCODE_IMM);
   wire is_load = (opcode == `OPCODE_LOAD);
   wire is_store = (opcode == `OPCODE_STORE);
   wire is_branch = (opcode == `OPCODE_BRANCH);
   wire is_lui = (opcode == `OPCODE_LUI);
   wire is_auipc = (opcode == `OPCODE_AUIPC);
   wire is_jal = (opcode == `OPCODE_JAL);
   wire is_jalr = (opcode == `OPCODE_JALR);

   // R-Type (ALU ops)
   wire is_add = is_rtype & (funct3 == `FUNC_ADD_SUB) & (funct7 == `AUX_FUNC_ADD);
   wire is_sub = is_rtype & (funct3 == `FUNC_ADD_SUB) & (funct7 == `AUX_FUNC_SUB);
   wire is_sll = is_rtype & (funct3 == `FUNC_SLL);
   wire is_slt = is_rtype & (funct3 == `FUNC_SLT);
   wire is_sltu = is_rtype & (funct3 == `FUNC_SLTU);
   wire is_xor = is_rtype & (funct3 == `FUNC_XOR);
   wire is_srl = is_rtype & (funct3 == `FUNC_SRL_SRA) & (funct7 == `AUX_FUNC_ADD);
   wire is_sra = is_rtype & (funct3 == `FUNC_SRL_SRA) & (funct7 == `AUX_FUNC_SUB);
   wire is_or = is_rtype & (funct3 == `FUNC_OR);
   wire is_and = is_rtype & (funct3 == `FUNC_AND);

   // I-Type (ALU ops)
   wire is_addi = is_itype & (funct3 == `FUNC_ADD_SUB);
   wire is_slti = is_itype & (funct3 == `FUNC_SLT);
   wire is_sltiu = is_itype & (funct3 == `FUNC_SLTU);
   wire is_xori = is_itype & (funct3 == `FUNC_XOR);
   wire is_ori  = is_itype & (funct3 == `FUNC_OR);
   wire is_andi = is_itype & (funct3 == `FUNC_AND);
   wire is_slli = is_itype & (funct3 == `FUNC_SLL);
   wire is_srli = is_itype & (funct3 == `FUNC_SRL_SRA) & (funct7 == `AUX_FUNC_ADD);
   wire is_srai = is_itype & (funct3 == `FUNC_SRL_SRA) & (funct7 == `AUX_FUNC_SUB);

   // Loads + Stores 
   wire is_lb = is_load  & (funct3 == `FUNC_LB_SB);
   wire is_lh = is_load  & (funct3 == `FUNC_LH_SH);
   wire is_lw = is_load  & (funct3 == `FUNC_LW_SW);
   wire is_lbu = is_load  & (funct3 == `FUNC_LBU);
   wire is_lhu = is_load  & (funct3 == `FUNC_LHU);
   wire is_sb = is_store & (funct3 == `FUNC_LB_SB);
   wire is_sh = is_store & (funct3 == `FUNC_LH_SH);
   wire is_sw = is_store & (funct3 == `FUNC_LW_SW);
   wire is_any_load = is_lb | is_lh | is_lw | is_lbu | is_lhu;

   // Branches
   wire is_beq = is_branch & (funct3 == `FUNC_BEQ);
   wire is_bne = is_branch & (funct3 == `FUNC_BNE);
   wire is_blt = is_branch & (funct3 == `FUNC_BLT);
   wire is_bge = is_branch & (funct3 == `FUNC_BGE);
   wire is_bltu = is_branch & (funct3 == `FUNC_BLTU);
   wire is_bgeu = is_branch & (funct3 == `FUNC_BGEU);

   // M-Extension
   wire is_mul = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_MUL);
   wire is_mulh = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_MULH);
   wire is_mulhsu = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_MULHSU);
   wire is_mulhu = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_MULHU);
   wire is_div = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_DIV);
   wire is_divu = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_DIVU);
   wire is_rem = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_REM);
   wire is_remu = is_rtype & (funct7 == `M_FUNC7) & (funct3 == `FUNC_REMU);

   // ID/EX: ALU operand selection
   // opA = rs1 (from RF); opB = rs2 or imm
   // for stores, addr = rs1 + immS
   // for loads/I-ops, = rs1 + immI

   wire ALUSrc = is_addi | is_slti | is_sltiu | is_xori | is_ori | is_andi |
                 is_slli | is_srli | is_srai |
                 is_any_load | is_sb | is_sh | is_sw;

   wire [`WORD_WIDTH-1:0] opB =
                  is_store ? immS :    // store address uses S-imm
                  ALUSrc   ? immI :    // I-ALU/loads use I-imm
                  Rdata2;   // otherwise use rs2
   
   // Keep funct7 for I-type shifts so SRLI ≠ SRAI
   wire [2:0] ALU_func = funct3;
   wire [6:0] ALU_auxFunc = (is_itype && (funct3==`FUNC_SLL || funct3==`FUNC_SRL_SRA)) ? funct7 :
                              is_rtype ? funct7 : 7'b0000000;
   

   // Execution Unit (EX) - ALU/shift/M
   // Produces ALU result used for arithmetic/logic, effective addr for load + stores,
   // and branch target adders use PC + immB

   wire [`WORD_WIDTH-1:0] ALUResult;

   ExecutionUnit EU(
      .out(ALUResult),
      .opA(Rdata1),        // rs1
      .opB(opB),           // rs2 or immediate
      .func(ALU_func),
      .auxFunc(ALU_auxFunc)
   );

   // Data Memory Access (MEM)
   // DataAddr = effective address from ALU
   // StoreData = rs2 (value to write on stores)
   // MemSize selects byte/half/word at memory

   assign DataAddr  = ALUResult;
   assign StoreData = Rdata2;

   assign MemSize = (is_lb | is_lbu | is_sb) ? `SIZE_BYTE :
                    (is_lh | is_lhu | is_sh) ? `SIZE_HWORD :
                                               `SIZE_WORD;

   // MEM --> WB (Load Extender)
   // Extract bytes based on address alignment
   wire [7:0]  byte0 = DataWord[7:0];
   wire [7:0]  byte1 = DataWord[15:8];
   wire [7:0]  byte2 = DataWord[23:16];
   wire [7:0]  byte3 = DataWord[31:24];
   wire [15:0] hword0 = DataWord[15:0];
   wire [15:0] hword1 = DataWord[31:16];

   // Select correct byte based on address alignment
   wire [7:0]  loaded_byte =  (DataAddr[1:0]==2'b00) ? byte0 :
                              (DataAddr[1:0]==2'b01) ? byte1 :
                              (DataAddr[1:0]==2'b10) ? byte2 : byte3;
   wire [15:0] loaded_hword = DataAddr[1] ? hword1 : hword0;

   wire [`WORD_WIDTH-1:0] extended_byte = is_lbu ? {24'b0, loaded_byte} :
                                           {{24{loaded_byte[7]}}, loaded_byte};
   wire [`WORD_WIDTH-1:0] extended_hword = is_lhu ? {16'b0, loaded_hword} :
                                           {{16{loaded_hword[15]}}, loaded_hword};

   wire [`WORD_WIDTH-1:0] LoadData = (is_lb | is_lbu) ? extended_byte :
                                     (is_lh | is_lhu) ? extended_hword :
                                                        DataWord; // lw

   
   // Writeback mux to rd (WB)
   wire [`WORD_WIDTH-1:0] auipc_result = PC + immU;
   wire [`WORD_WIDTH-1:0] WB_data = is_any_load ? LoadData : ALUResult;

   wire [`WORD_WIDTH-1:0] RWrdata_pre = (is_jal | is_jalr) ? PC_Plus_4 :
                                         is_lui ? immU :
                                         is_auipc ? auipc_result :
                                       WB_data;
   wire rd_is_x0 = (Rdst == 5'd0);


   // IF/ID/EX: Branch Compare
   wire signed_lt = ($signed(Rdata1) < $signed(Rdata2));
   wire unsigned_lt = (Rdata1 < Rdata2);

   wire take_branch = (is_beq  & (Rdata1 == Rdata2)) | (is_bne  & (Rdata1 != Rdata2)) |
                      (is_blt  &  signed_lt) | (is_bge  & ~signed_lt) | (is_bltu &  unsigned_lt) |
                      (is_bgeu & ~unsigned_lt);

   wire [`WORD_WIDTH-1:0] BranchTarget = PC + immB;
   wire [`WORD_WIDTH-1:0] JalTarget = PC + immJ;
   wire [`WORD_WIDTH-1:0] JalrTarget = (Rdata1 + immI) & ~32'b1;

   wire [`WORD_WIDTH-1:0] NPC =  is_jal   ? JalTarget :
                                 is_jalr  ? JalrTarget :
                                 take_branch ? BranchTarget :
                              PC_Plus_4;
   

   // Misalignment + Illegal Detection
   wire fetch_misalign = |PC[1:0];
   wire hword_misalign = (is_lh | is_lhu | is_sh) & DataAddr[0];
   wire word_misalign = (is_lw | is_sw) & |DataAddr[1:0];
   wire data_misalign = hword_misalign | word_misalign;

   wire illegal_op = ~ (is_add | is_sub | is_sll | is_slt | is_sltu | is_xor |
                       is_srl | is_sra | is_or | is_and |
                       is_addi | is_slti | is_sltiu | is_xori | is_ori | is_andi |
                       is_slli | is_srli | is_srai |
                       is_lb | is_lh | is_lw | is_lbu | is_lhu | is_sb | is_sh | is_sw |
                       is_beq | is_bne | is_blt | is_bge | is_bltu | is_bgeu |
                       is_mul | is_mulh | is_mulhsu | is_mulhu | is_div | is_divu | is_rem | is_remu |
                       is_lui | is_auipc | is_jal | is_jalr);

   assign halt = fetch_misalign | data_misalign | illegal_op;

   // Write enables (CNTRL)
   wire MemWrEn_pre = is_sb | is_sh | is_sw;
   wire RWrEn_pre = (is_rtype | is_itype | is_any_load | is_lui | is_auipc | is_jal | is_jalr |
                      (is_mul | is_mulh | is_mulhsu | is_mulhu | is_div | is_divu | is_rem | is_remu));

   assign MemWrEn = MemWrEn_pre & ~halt;
   assign RWrEn = RWrEn_pre   & ~halt & ~rd_is_x0;
   assign RWrdata = RWrdata_pre;

   // Instruction Memory (IF) - System State
   Mem   MEM(.InstAddr(PC), .InstOut(InstWord), 
            .DataAddr(DataAddr), .DataSize(MemSize), .DataIn(StoreData), .DataOut(DataWord), .WE(MemWrEn), .CLK(clk));

   RegFile RF(.AddrA(Rsrc1), .DataOutA(Rdata1),
	         .AddrB(Rsrc2), .DataOutB(Rdata2), 
	         .AddrW(Rdst), .DataInW(RWrdata), .WenW(RWrEn), .CLK(clk));

   Reg PC_REG(.Din(NPC), .Qout(PC), .WE(~halt), .CLK(clk), .RST(rst));
   
endmodule // SingleCycleCPU


// Incomplete version of Lab2 execution unit
// You will need to extend it. Feel free to modify the interface also

module ExecutionUnit(out, opA, opB, func, auxFunc);
   output [`WORD_WIDTH-1:0] out;
   input [`WORD_WIDTH-1:0]  opA, opB;
   input [2:0] 	 func;
   input [6:0] 	 auxFunc;

   // Shifts
   wire [4:0] shamt = opB[4:0];

   // Signed/Unsigned
   wire signed [31:0] opA_s = opA;
   wire signed [31:0] opB_s = opB;

   // Base ALU ops
   wire [31:0] add = opA + opB;
   wire [31:0] sub = opA - opB;
   wire [31:0] _and = opA & opB;
   wire [31:0] _or = opA | opB;
   wire [31:0] _xor = opA ^ opB;
   wire [31:0] sll = opA << shamt;
   wire [31:0] srl = opA >> shamt;
   wire [31:0] sra = opA_s >>> shamt;
   wire [31:0] slt = (opA_s < opB_s) ? 32'd1 : 32'd0;
   wire [31:0] sltu = (opA < opB  ) ? 32'd1 : 32'd0;

   // Multiplication
   // signed * signed
   wire [63:0] mul_ss = $signed(opA) * $signed(opB);
   // unsigned * unsigned
   wire [63:0] mul_uu = $unsigned(opA) * $unsigned(opB);
   // signed * unsigned
   wire [63:0] mul_su = $signed(opA) * $unsigned(opB);

   // Division
   wire div_by_zero  = (opB == 32'b0);
   wire div_overflow = (opA == 32'h8000_0000) && (opB == 32'hFFFF_FFFF);

   wire [31:0] div_s_q = div_by_zero  ? 32'hFFFF_FFFF :
                        div_overflow ? 32'h8000_0000 :
                        (opA_s / opB_s);

   wire [31:0] div_u_q = div_by_zero  ? 32'hFFFF_FFFF :
                        (opA / opB);

   wire [31:0] rem_s_r = div_by_zero  ? opA :
                        div_overflow ? 32'h0000_0000 :
                        (opA_s % opB_s);

   wire [31:0] rem_u_r = div_by_zero  ? opA :
                        (opA % opB);


   // Result Selection
   assign out =
      (auxFunc==7'b0000000 && func==3'b000) ? add :
      (auxFunc==7'b0000000 && func==3'b001) ? sll :
      (auxFunc==7'b0000000 && func==3'b010) ? slt :
      (auxFunc==7'b0000000 && func==3'b011) ? sltu :
      (auxFunc==7'b0000000 && func==3'b100) ? _xor :
      (auxFunc==7'b0000000 && func==3'b101) ? srl :
      (auxFunc==7'b0000000 && func==3'b110) ? _or :
      (auxFunc==7'b0000000 && func==3'b111) ? _and :
      (auxFunc==7'b0100000 && func==3'b000) ? sub : 
      (auxFunc==7'b0100000 && func==3'b101) ? sra :
      (auxFunc==7'b0000001 && func==3'b000) ? mul_ss[31:0] : 
      (auxFunc==7'b0000001 && func==3'b001) ? mul_ss[63:32] :
      (auxFunc==7'b0000001 && func==3'b010) ? mul_su[63:32] :
      (auxFunc==7'b0000001 && func==3'b011) ? mul_uu[63:32] :
      (auxFunc==7'b0000001 && func==3'b100) ? div_s_q :
      (auxFunc==7'b0000001 && func==3'b101) ? div_u_q :
      (auxFunc==7'b0000001 && func==3'b110) ? rem_s_r :
      (auxFunc==7'b0000001 && func==3'b111) ? rem_u_r :
      
      // default - illegal
      32'b0;
   
endmodule // ExecutionUnit
