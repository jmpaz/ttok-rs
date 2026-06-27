{
  description = "ttok-rs: fast token counter using tiktoken-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems =
        f:
        lib.genAttrs systems (
          system:
          let
            pkgs = import nixpkgs { inherit system; };
          in
          f pkgs
        );
    in
    {
      packages = forAllSystems (pkgs: {
        default = pkgs.rustPlatform.buildRustPackage {
          pname = "ttok-rs";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          meta = {
            description = "Fast token counter using tiktoken-rs";
            homepage = "https://github.com/jmpaz/ttok-rs";
            license = lib.licenses.mit;
            mainProgram = "ttok-rs";
            platforms = lib.platforms.unix;
          };
        };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          inputsFrom = [ self.packages.${pkgs.system}.default ];
          packages = [
            pkgs.cargo
            pkgs.clippy
            pkgs.rust-analyzer
            pkgs.rustc
            pkgs.rustfmt
          ];
        };
      });

      apps = forAllSystems (pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.system}.default}/bin/ttok-rs";
        };
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt-rfc-style);
    };
}
